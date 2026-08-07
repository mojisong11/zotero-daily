from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


WEEKDAY_ALIASES = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


def normalize_schedule_weekdays(weekdays: list[str] | ListConfig | None, source: str) -> set[int] | None:
    if weekdays is None:
        return None

    if not isinstance(weekdays, (list, ListConfig)):
        raise TypeError(
            f"config.source.{source}.schedule.weekdays must be a list of weekday strings or null, "
            'for example ["mon","wed"].'
        )

    normalized: set[int] = set()
    for weekday in weekdays:
        if not isinstance(weekday, str):
            raise TypeError(f"config.source.{source}.schedule.weekdays must contain only weekday strings.")
        weekday_key = weekday.strip().lower()
        if weekday_key not in WEEKDAY_ALIASES:
            raise ValueError(
                f"config.source.{source}.schedule.weekdays contains unknown weekday '{weekday}'. "
                "Use values like mon, tue, wed, thu, fri, sat, sun."
            )
        normalized.add(WEEKDAY_ALIASES[weekday_key])

    if len(normalized) == 0:
        raise ValueError(f"config.source.{source}.schedule.weekdays must not be empty.")

    return normalized


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    def get_max_paper_num_for_source(self, source: str) -> int:
        if source == "arxiv":
            return self.config.executor.max_paper_num

        source_limit = self.config.executor.get(f"max_{source}_num")
        if source_limit is None:
            return self.config.executor.max_paper_num
        return source_limit

    def should_run_source_today(self, source: str, now: datetime | None = None) -> bool:
        source_config = getattr(self.config.source, source)
        schedule = source_config.get("schedule")
        if schedule is None:
            return True

        weekdays = normalize_schedule_weekdays(schedule.get("weekdays"), source)
        if weekdays is None:
            return True

        timezone_name = schedule.get("timezone") or "UTC"
        try:
            timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                f"config.source.{source}.schedule.timezone '{timezone_name}' is not a valid IANA timezone."
            ) from exc

        current_time = now or datetime.now(timezone)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone)
        else:
            current_time = current_time.astimezone(timezone)

        return current_time.weekday() in weekdays

    
    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return
        papers_by_source = {source: [] for source in self.retrievers}
        for source, retriever in self.retrievers.items():
            if not self.should_run_source_today(source):
                schedule = getattr(self.config.source, source).get("schedule")
                timezone_name = schedule.get("timezone") or "UTC"
                weekdays = schedule.get("weekdays")
                logger.info(
                    f"Skipping {source} today because it is scheduled only for {list(weekdays)} in timezone {timezone_name}"
                )
                continue
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            papers_by_source[source] = papers
        total_paper_num = sum(len(papers) for papers in papers_by_source.values())
        logger.info(f"Total {total_paper_num} papers retrieved from all sources")
        reranked_papers_by_source = {source: [] for source in papers_by_source}
        selected_papers = []
        if total_paper_num > 0:
            logger.info("Reranking papers...")
            for source, papers in papers_by_source.items():
                if len(papers) == 0:
                    continue
                logger.info(f"Reranking {source} papers...")
                reranked_papers = self.reranker.rerank(papers, corpus)
                max_paper_num = self.get_max_paper_num_for_source(source)
                if max_paper_num != -1:
                    reranked_papers = reranked_papers[:max_paper_num]
                logger.info(f"Selected {len(reranked_papers)} {source} papers after reranking")
                reranked_papers_by_source[source] = reranked_papers
                selected_papers.extend(reranked_papers)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        if len(selected_papers) > 0:
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(selected_papers, desc="Generating paper summaries"):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
        logger.info("Sending email...")
        email_content = render_email(reranked_papers_by_source)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
