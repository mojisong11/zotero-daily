import arxiv
import argparse
import os
import sys
import time
import random
import yaml
from dotenv import load_dotenv

load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm import tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper, BiorxivPaper
from llm import set_global_llm
import feedparser
from datetime import datetime, timedelta
import requests


# ----------------------------
# 通用工具
# ----------------------------

def chunked(items, size):
    """将列表切分为固定大小的小批次。"""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def is_retryable_arxiv_error(exc: Exception) -> bool:
    """
    判断是否属于适合重试的 arXiv 网络错误。
    429: 请求过多
    500/502/503/504: arXiv 服务端临时异常
    """
    text = str(exc)
    retryable_codes = ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504")
    return any(code in text for code in retryable_codes)


# ----------------------------
# Zotero
# ----------------------------

def get_zotero_corpus(id: str, key: str) -> list[dict]:
    zot = zotero.Zotero(id, "user", key)

    collections = zot.everything(zot.collections())
    collections = {c["key"]: c for c in collections}

    corpus = zot.everything(
        zot.items(itemType="conferencePaper || journalArticle || preprint")
    )
    corpus = [c for c in corpus if c["data"]["abstractNote"] != ""]

    def get_collection_path(col_key: str) -> str:
        parent = collections[col_key]["data"]["parentCollection"]
        if parent:
            return (
                get_collection_path(parent)
                + "/"
                + collections[col_key]["data"]["name"]
            )
        return collections[col_key]["data"]["name"]

    for c in corpus:
        paths = [
            get_collection_path(col)
            for col in c["data"]["collections"]
            if col in collections
        ]
        c["paths"] = paths

    return corpus


def filter_corpus(corpus: list[dict], pattern: str) -> list[dict]:
    fd, filename = mkstemp()
    os.close(fd)

    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(pattern)

        matcher = parse_gitignore(filename, base_dir="./")
        new_corpus = []

        for c in corpus:
            match_results = [matcher(p) for p in c["paths"]]
            if not any(match_results):
                new_corpus.append(c)

        return new_corpus
    finally:
        if os.path.exists(filename):
            os.remove(filename)


# ----------------------------
# arXiv
# ----------------------------

def fetch_arxiv_batch(
    client: arxiv.Client,
    paper_ids: list[str],
    batch_index: int,
    max_attempts: int = 5,
) -> list[ArxivPaper]:
    """
    获取一个小批次 arXiv 论文。

    对 429/5xx 使用指数退避重试：
    30s -> 60s -> 120s -> 240s ...
    """
    search = arxiv.Search(id_list=paper_ids)

    for attempt in range(1, max_attempts + 1):
        try:
            results = list(client.results(search))
            return [ArxivPaper(p) for p in results]

        except Exception as exc:
            if not is_retryable_arxiv_error(exc):
                raise

            if attempt == max_attempts:
                logger.error(
                    f"arXiv batch {batch_index} failed after "
                    f"{max_attempts} attempts: {exc}"
                )
                return []

            base_wait = 30 * (2 ** (attempt - 1))
            jitter = random.randint(0, 10)
            wait_seconds = base_wait + jitter

            logger.warning(
                f"arXiv batch {batch_index} received a temporary error "
                f"(attempt {attempt}/{max_attempts}): {exc}"
            )
            logger.warning(
                f"Waiting {wait_seconds} seconds before retrying..."
            )
            time.sleep(wait_seconds)

    return []


def get_arxiv_paper(query: str, debug: bool = False) -> list[ArxivPaper]:
    """
    从 arXiv RSS 获取当天新论文 ID，再通过 arXiv API 获取完整元数据。

    为避免 GitHub Actions 共享出口 IP 触发 HTTP 429：
    - 每批只请求 10 篇；
    - API 请求之间至少间隔 5 秒；
    - 对 429/503 等临时错误进行指数退避；
    - 单个批次失败时跳过该批次，而不是让整个邮件任务退出。
    """
    if debug:
        logger.debug("Retrieve 5 arXiv papers regardless of the date.")

        client = arxiv.Client(
            page_size=5,
            delay_seconds=5.0,
            num_retries=3,
        )
        search = arxiv.Search(
            query="cat:cs.AI",
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        papers = []
        for result in client.results(search):
            papers.append(ArxivPaper(result))
            if len(papers) == 5:
                break

        return papers

    if not query:
        logger.warning("ARXIV_QUERY is empty. Skipping arXiv retrieval.")
        return []

    feed_url = f"https://rss.arxiv.org/atom/{query}"
    logger.info(f"Retrieving arXiv RSS feed from {feed_url}...")

    feed = feedparser.parse(feed_url)

    if getattr(feed, "bozo", False):
        logger.warning(
            f"arXiv RSS parsing warning: {getattr(feed, 'bozo_exception', 'unknown')}"
        )

    feed_title = getattr(feed.feed, "title", "")
    if "Feed error for query" in feed_title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")

    all_paper_ids = []
    for entry in feed.entries:
        announce_type = getattr(entry, "arxiv_announce_type", "")
        entry_id = getattr(entry, "id", "")

        if announce_type == "new" and entry_id:
            paper_id = entry_id.removeprefix("oai:arXiv.org:")
            all_paper_ids.append(paper_id)

    # 去重但保持原始顺序
    all_paper_ids = list(dict.fromkeys(all_paper_ids))

    if not all_paper_ids:
        logger.info("No new arXiv papers found.")
        return []

    logger.info(f"Found {len(all_paper_ids)} new arXiv paper IDs.")

    # page_size 与 batch_size 都设小一些，减少 429 风险
    batch_size = 10
    client = arxiv.Client(
        page_size=batch_size,
        delay_seconds=5.0,
        num_retries=2,
    )

    papers = []
    batches = list(chunked(all_paper_ids, batch_size))

    bar = tqdm(
        total=len(all_paper_ids),
        desc="Retrieving Arxiv papers",
    )

    try:
        for batch_index, paper_id_batch in enumerate(batches, start=1):
            batch = fetch_arxiv_batch(
                client=client,
                paper_ids=paper_id_batch,
                batch_index=batch_index,
                max_attempts=5,
            )

            papers.extend(batch)
            bar.update(len(paper_id_batch))

            logger.info(
                f"arXiv batch {batch_index}/{len(batches)}: "
                f"requested {len(paper_id_batch)}, retrieved {len(batch)}."
            )

            # 除 arxiv.Client 自身 delay 外，再在批次间主动等待
            if batch_index < len(batches):
                time.sleep(5)

    finally:
        bar.close()

    missing_count = len(all_paper_ids) - len(papers)
    if missing_count > 0:
        logger.warning(
            f"Retrieved {len(papers)}/{len(all_paper_ids)} arXiv papers. "
            f"{missing_count} papers were skipped because arXiv kept "
            f"returning temporary errors."
        )

    return papers


# ----------------------------
# bioRxiv
# ----------------------------

def get_biorxiv_paper(query: str, debug: bool = False) -> list[BiorxivPaper]:
    if debug:
        url = (
            "https://api.biorxiv.org/details/biorxiv/"
            "2025-03-21/2025-03-28?category=cell_biology"
        )

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        data = response.json()
        logger.debug("Retrieve 5 bioRxiv papers regardless of the date.")

        papers = []
        for item in data.get("collection", []):
            if item.get("doi", "") == "":
                continue

            papers.append(BiorxivPaper(item))
            if len(papers) == 5:
                break

        return papers

    if not query:
        logger.warning("BIORXIV_QUERY is empty. Skipping bioRxiv retrieval.")
        return []

    today = datetime.now()
    yesterday = today - timedelta(days=1)

    formatted_date = today.strftime("%Y-%m-%d")
    formatted_yesterday = yesterday.strftime("%Y-%m-%d")

    queries = query.split("+") if "+" in query else [query]
    papers = []

    for category in queries:
        category = category.strip()
        if not category:
            continue

        url = (
            f"https://api.biorxiv.org/details/biorxiv/"
            f"{formatted_yesterday}/{formatted_date}?category={category}"
        )
        logger.info(f"Retrieving bioRxiv papers from {url}...")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error(
                f"Failed to retrieve bioRxiv category '{category}': {exc}"
            )
            # 一个 bioRxiv 分类失败时继续处理其他分类
            continue

        data = response.json()

        for item in data.get("collection", []):
            if item.get("doi", "") == "":
                continue

            papers.append(BiorxivPaper(item))

    return papers


# ----------------------------
# 参数
# ----------------------------

parser = argparse.ArgumentParser(
    description="Recommender system for academic papers"
)


def add_argument(*args, **kwargs):
    def get_env(key: str, default=None):
        # Workflow 中未设置的环境变量可能会以空字符串传入
        value = os.environ.get(key)
        if value == "" or value is None:
            return default
        return value

    parser.add_argument(*args, **kwargs)

    arg_full_name = kwargs.get("dest", args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)

    if env_value is not None:
        arg_type = kwargs.get("type")

        if arg_type == bool:
            env_value = env_value.lower() in ["true", "1", "yes", "y"]
        elif arg_type is not None:
            env_value = arg_type(env_value)

        parser.set_defaults(**{arg_full_name: env_value})


# ----------------------------
# 主程序
# ----------------------------

if __name__ == "__main__":
    add_argument("--zotero_id", type=str, help="Zotero user ID")
    add_argument("--zotero_key", type=str, help="Zotero API key")
    add_argument(
        "--zotero_ignore",
        type=str,
        help="Zotero collection to ignore, using gitignore-style pattern.",
    )
    add_argument(
        "--send_empty",
        type=bool,
        help="If no arXiv or bioRxiv paper is found, send an empty email",
        default=False,
    )
    add_argument(
        "--max_paper_num",
        type=int,
        help="Maximum number of arXiv papers to recommend",
        default=50,
    )
    add_argument(
        "--max_biorxiv_num",
        type=int,
        help="Maximum number of bioRxiv papers to recommend",
        default=50,
    )
    add_argument("--arxiv_query", type=str, help="Arxiv search query")
    add_argument("--biorxiv_query", type=str, help="Biorxiv search category")
    add_argument("--smtp_server", type=str, help="SMTP server")
    add_argument("--smtp_port", type=int, help="SMTP port")
    add_argument("--sender", type=str, help="Sender email address")
    add_argument("--receiver", type=str, help="Receiver email address")
    add_argument(
        "--sender_password",
        type=str,
        help="Sender email password or authorization code",
    )
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI-compatible API to generate TLDR",
        default=False,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM model name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )

    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    # 从 config.yml 覆盖命令行/环境变量参数
    if os.path.exists("config.yml"):
        with open("config.yml", "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    if args.use_llm_api and args.openai_api_key is None:
        raise ValueError(
            "use_llm_api=True, but OPENAI_API_KEY/openai_api_key is not set."
        )

    logger.remove()
    logger.add(sys.stdout, level="DEBUG" if args.debug else "INFO")

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")

    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n{args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")

    # arXiv 失败时不再让整个任务中止
    logger.info("Retrieving arXiv papers...")
    try:
        papers = get_arxiv_paper(args.arxiv_query, args.debug)
    except Exception:
        logger.exception(
            "Unexpected error while retrieving arXiv papers. "
            "Continuing with an empty arXiv list."
        )
        papers = []

    # bioRxiv 失败时也不让整个任务中止
    logger.info("Retrieving bioRxiv papers...")
    try:
        biorxiv_papers = get_biorxiv_paper(
            args.biorxiv_query,
            args.debug,
        )
    except Exception:
        logger.exception(
            "Unexpected error while retrieving bioRxiv papers. "
            "Continuing with an empty bioRxiv list."
        )
        biorxiv_papers = []

    logger.info(f"Retrieved {len(papers)} papers from arXiv.")
    logger.info(f"Retrieved {len(biorxiv_papers)} papers from bioRxiv.")

    if len(papers) == 0 and len(biorxiv_papers) == 0:
        logger.info(
            "No new papers found. Yesterday may have been a holiday, "
            "or the upstream APIs may have been temporarily unavailable."
        )

        if not args.send_empty:
            sys.exit(0)

    else:
        logger.info("Reranking papers...")
        papers, biorxiv_papers = rerank_paper(
            papers,
            biorxiv_papers,
            corpus,
        )

        if (
            args.max_paper_num != -1
            and args.max_paper_num < len(papers)
        ):
            papers = papers[:args.max_paper_num]

        if (
            args.max_biorxiv_num != -1
            and args.max_biorxiv_num < len(biorxiv_papers)
        ):
            biorxiv_papers = biorxiv_papers[:args.max_biorxiv_num]

        if args.use_llm_api:
            logger.info("Using OpenAI-compatible API as global LLM.")
            set_global_llm(
                api_key=args.openai_api_key,
                base_url=args.openai_api_base,
                model=args.model_name,
                lang=args.language,
            )
        else:
            logger.info("Using Local LLM as global LLM.")
            set_global_llm(lang=args.language)

    html = render_email(papers, biorxiv_papers)

    logger.info("Sending email...")
    send_email(
        args.sender,
        args.receiver,
        args.sender_password,
        args.smtp_server,
        args.smtp_port,
        html,
    )

    logger.success(
        "Email sent successfully! If you do not receive it, "
        "check the configuration and junk folder."
    )
