from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import unescape
import json
import re
from time import sleep
from typing import Any
from urllib.parse import quote

import requests
from loguru import logger
from omegaconf import DictConfig, ListConfig

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


DOI_PATTERN = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
TITLE_EXCLUDE_PREFIXES = (
    "author correction",
    "publisher correction",
    "correction",
    "erratum",
    "corrigendum",
    "retraction",
    "editorial",
    "news",
)
DEFAULT_USER_AGENT = "zotero-arxiv-daily/1.0"
DEFAULT_DAYS = 2
DEFAULT_ROWS_PER_QUERY = 100
RETRY_STATUSES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class JournalSpec:
    label: str
    query_titles: list[str]
    match_titles: list[str]


DEFAULT_SUBJOURNAL_SPECS: dict[str, JournalSpec] = {
    "Nature Biotechnology": JournalSpec(
        label="Nature Biotechnology",
        query_titles=["Nature Biotechnology"],
        match_titles=["Nature Biotechnology"],
    ),
    "Nature Chemical Biology": JournalSpec(
        label="Nature Chemical Biology",
        query_titles=["Nature Chemical Biology"],
        match_titles=["Nature Chemical Biology"],
    ),
    "Nature Methods": JournalSpec(
        label="Nature Methods",
        query_titles=["Nature Methods"],
        match_titles=["Nature Methods"],
    ),
    "Nature Machine Intelligence": JournalSpec(
        label="Nature Machine Intelligence",
        query_titles=["Nature Machine Intelligence"],
        match_titles=["Nature Machine Intelligence"],
    ),
    "Nature Computational Science": JournalSpec(
        label="Nature Computational Science",
        query_titles=["Nature Computational Science"],
        match_titles=["Nature Computational Science"],
    ),
    "Nature Communications": JournalSpec(
        label="Nature Communications",
        query_titles=["Nature Communications"],
        match_titles=["Nature Communications"],
    ),
    "Cell Chemical Biology": JournalSpec(
        label="Cell Chemical Biology",
        query_titles=["Cell Chemical Biology"],
        match_titles=["Cell Chemical Biology"],
    ),
    "Cell Reports Methods": JournalSpec(
        label="Cell Reports Methods",
        query_titles=["Cell Reports Methods"],
        match_titles=["Cell Reports Methods"],
    ),
    "Science Advances": JournalSpec(
        label="Science Advances",
        query_titles=["Science Advances"],
        match_titles=["Science Advances"],
    ),
    "PNAS": JournalSpec(
        label="PNAS",
        query_titles=["Proceedings of the National Academy of Sciences"],
        match_titles=[
            "PNAS",
            "Proceedings of the National Academy of Sciences",
            "Proceedings of the National Academy of Sciences of the United States of America",
        ],
    ),
    "Journal of the American Chemical Society": JournalSpec(
        label="Journal of the American Chemical Society",
        query_titles=["Journal of the American Chemical Society"],
        match_titles=["Journal of the American Chemical Society", "JACS"],
    ),
    "JACS Au": JournalSpec(
        label="JACS Au",
        query_titles=["JACS Au"],
        match_titles=["JACS Au"],
    ),
    "ACS Central Science": JournalSpec(
        label="ACS Central Science",
        query_titles=["ACS Central Science"],
        match_titles=["ACS Central Science"],
    ),
    "ACS Chemical Biology": JournalSpec(
        label="ACS Chemical Biology",
        query_titles=["ACS Chemical Biology"],
        match_titles=["ACS Chemical Biology"],
    ),
    "Journal of Medicinal Chemistry": JournalSpec(
        label="Journal of Medicinal Chemistry",
        query_titles=["Journal of Medicinal Chemistry"],
        match_titles=["Journal of Medicinal Chemistry"],
    ),
    "Chemical Science": JournalSpec(
        label="Chemical Science",
        query_titles=["Chemical Science"],
        match_titles=["Chemical Science"],
    ),
    "Angewandte Chemie": JournalSpec(
        label="Angewandte Chemie",
        query_titles=["Angewandte Chemie", "Angewandte Chemie International Edition"],
        match_titles=["Angewandte Chemie", "Angewandte Chemie International Edition"],
    ),
    "Nucleic Acids Research": JournalSpec(
        label="Nucleic Acids Research",
        query_titles=["Nucleic Acids Research"],
        match_titles=["Nucleic Acids Research"],
    ),
    "Bioinformatics": JournalSpec(
        label="Bioinformatics",
        query_titles=["Bioinformatics"],
        match_titles=["Bioinformatics"],
    ),
    "Briefings in Bioinformatics": JournalSpec(
        label="Briefings in Bioinformatics",
        query_titles=["Briefings in Bioinformatics"],
        match_titles=["Briefings in Bioinformatics"],
    ),
}


def normalize_title(title: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", title).strip().lower()


def clean_markup(text: str | None) -> str | None:
    if not text:
        return None

    cleaned = unescape(text)
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned or None


def split_author_string(author_string: str | None) -> list[str]:
    if not author_string:
        return []
    if ";" in author_string:
        return [author.strip() for author in author_string.split(";") if author.strip()]
    if "," in author_string:
        return [author.strip() for author in author_string.split(",") if author.strip()]
    return [author_string.strip()]


@register_retriever("subjournal")
class SubjournalRetriever(BaseRetriever):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.journal_specs = self._resolve_journal_specs(self.retriever_config.get("journals"))
        if len(self.journal_specs) == 0:
            raise ValueError("journals must be specified for subjournal")

    def _resolve_journal_specs(self, journal_configs: list[Any] | ListConfig | None) -> list[JournalSpec]:
        if journal_configs is None:
            return []

        specs: list[JournalSpec] = []
        for entry in journal_configs:
            if isinstance(entry, str):
                default_spec = DEFAULT_SUBJOURNAL_SPECS.get(entry)
                if default_spec is not None:
                    specs.append(default_spec)
                else:
                    specs.append(JournalSpec(label=entry, query_titles=[entry], match_titles=[entry]))
                continue

            if isinstance(entry, (dict, DictConfig)):
                label = str(entry.get("label") or entry.get("name"))
                if not label:
                    raise ValueError("Each subjournal config object must contain label or name.")
                query_titles = entry.get("query_titles") or entry.get("query_title") or [label]
                match_titles = entry.get("match_titles") or [label]
                if isinstance(query_titles, str):
                    query_titles = [query_titles]
                if isinstance(match_titles, str):
                    match_titles = [match_titles]
                specs.append(
                    JournalSpec(
                        label=label,
                        query_titles=[str(title) for title in query_titles],
                        match_titles=[str(title) for title in match_titles],
                    )
                )
                continue

            raise TypeError("source.subjournal.journals entries must be strings or mapping objects.")

        return specs

    def _request_json(self, url: str) -> dict[str, Any]:
        user_agent = self.retriever_config.get("user_agent") or DEFAULT_USER_AGENT
        mailto = self.retriever_config.get("mailto") or self.config.email.sender
        if mailto:
            user_agent = f"{user_agent} (mailto:{mailto})"

        max_attempts = 5
        timeout = 60
        headers = {"User-Agent": user_agent}
        for attempt in range(1, max_attempts + 1):
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code not in RETRY_STATUSES:
                response.raise_for_status()
                return response.json()

            if attempt == max_attempts:
                response.raise_for_status()

            wait_seconds = 10 * attempt
            logger.warning(f"Temporary HTTP {response.status_code} for {url}. Retry {attempt}/{max_attempts} in {wait_seconds}s.")
            sleep(wait_seconds)

        raise RuntimeError(f"Failed to retrieve {url}")

    def _get_date_range(self) -> tuple[str, str]:
        days = int(self.retriever_config.get("days") or DEFAULT_DAYS)
        end_days_ago = int(self.retriever_config.get("end_days_ago") or 0)
        if days < 1:
            raise ValueError("source.subjournal.days must be at least 1.")
        if end_days_ago < 0:
            raise ValueError("source.subjournal.end_days_ago must be non-negative.")
        today = datetime.now(timezone.utc).date()
        end_date = today - timedelta(days=end_days_ago)
        start_date = end_date - timedelta(days=days - 1)
        return start_date.isoformat(), end_date.isoformat()

    def _fetch_crossref_items(self, journal_spec: JournalSpec, start_date: str, end_date: str) -> list[dict[str, Any]]:
        rows = int(self.retriever_config.get("rows_per_query") or DEFAULT_ROWS_PER_QUERY)
        matched_items: list[dict[str, Any]] = []
        accepted_titles = {normalize_title(title) for title in journal_spec.match_titles}

        for query_title in journal_spec.query_titles:
            encoded_title = quote(query_title)
            url = (
                "https://api.crossref.org/works"
                f"?query.container-title={encoded_title}"
                f"&filter=from-pub-date:{start_date},until-pub-date:{end_date}"
                f"&rows={rows}"
            )
            payload = self._request_json(url)
            for item in payload.get("message", {}).get("items", []):
                container_titles = item.get("container-title") or []
                if len(container_titles) == 0:
                    continue
                journal_title = str(container_titles[0]).strip()
                if normalize_title(journal_title) not in accepted_titles:
                    continue
                if item.get("type") != "journal-article":
                    continue

                title_list = item.get("title") or []
                title = str(title_list[0]).strip() if title_list else ""
                if normalize_title(title).startswith(TITLE_EXCLUDE_PREFIXES):
                    continue

                matched_items.append(item)

        return matched_items

    def _fetch_europepmc_record(self, doi: str) -> dict[str, Any] | None:
        if not self.retriever_config.get("use_europepmc", True):
            return None

        query = quote(f'DOI:"{doi}"')
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={query}&format=json&resultType=core&pageSize=25"
        payload = self._request_json(url)
        results = payload.get("resultList", {}).get("result", [])
        if len(results) == 0:
            return None

        preferred_sources = {"PMC": 0, "MED": 1, "AGR": 2}

        def sort_key(result: dict[str, Any]) -> tuple[int, int]:
            source_rank = preferred_sources.get(str(result.get("source")), 99)
            has_abstract = 0 if result.get("abstractText") else 1
            return has_abstract, source_rank

        results = sorted(results, key=sort_key)
        return results[0]

    def _extract_crossref_article_url(self, item: dict[str, Any]) -> str:
        resource_url = item.get("resource", {}).get("primary", {}).get("URL")
        if resource_url:
            return str(resource_url)

        for link in item.get("link") or []:
            if link.get("content-type") == "text/html":
                return str(link.get("URL"))

        return str(item.get("URL") or "")

    def _extract_crossref_pdf_url(self, item: dict[str, Any]) -> str | None:
        for link in item.get("link") or []:
            if link.get("content-type") == "application/pdf":
                return str(link.get("URL"))
        return None

    def _fetch_article_page_abstract(self, article_url: str | None) -> str | None:
        if not article_url:
            return None

        try:
            response = requests.get(article_url, headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=60)
            response.raise_for_status()
        except Exception as exc:
            logger.debug(f"Failed to retrieve article page for abstract extraction: {article_url} ({exc})")
            return None

        content = response.text
        meta_patterns = [
            r'<meta[^>]+name=["\']citation_abstract["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']dc.description["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
            r'"description"\s*:\s*"([^"]+)"',
        ]
        for pattern in meta_patterns:
            match = re.search(pattern, content, flags=re.IGNORECASE)
            if match is not None:
                return clean_markup(match.group(1))
        return None

    def _pick_abstract(self, crossref_item: dict[str, Any], europepmc_record: dict[str, Any] | None, article_url: str) -> str | None:
        if europepmc_record is not None:
            europepmc_abstract = clean_markup(europepmc_record.get("abstractText"))
            if europepmc_abstract:
                return europepmc_abstract

        crossref_abstract = clean_markup(crossref_item.get("abstract"))
        if crossref_abstract:
            return crossref_abstract

        return self._fetch_article_page_abstract(article_url)

    def _extract_authors(self, crossref_item: dict[str, Any], europepmc_record: dict[str, Any] | None) -> list[str]:
        authors = []
        for author in crossref_item.get("author") or []:
            given = str(author.get("given") or "").strip()
            family = str(author.get("family") or "").strip()
            full_name = " ".join(part for part in [given, family] if part).strip()
            if full_name:
                authors.append(full_name)

        if len(authors) > 0:
            return authors

        if europepmc_record is not None:
            return split_author_string(europepmc_record.get("authorString"))

        return authors

    def _build_raw_entry(self, crossref_item: dict[str, Any]) -> dict[str, Any] | None:
        doi = crossref_item.get("DOI")
        if not doi:
            return None

        article_url = self._extract_crossref_article_url(crossref_item)
        europepmc_record = self._fetch_europepmc_record(str(doi))
        abstract = self._pick_abstract(crossref_item, europepmc_record, article_url)
        if not abstract:
            logger.info(f"Skipping {doi}: no abstract available from Europe PMC, Crossref, or article metadata.")
            return None

        container_titles = crossref_item.get("container-title") or []
        journal_title = str(container_titles[0]).strip() if container_titles else "Unknown Journal"
        title_list = crossref_item.get("title") or []
        title = str(title_list[0]).strip() if title_list else str(doi)
        pdf_url = self._extract_crossref_pdf_url(crossref_item) or article_url or f"https://doi.org/{doi}"
        authors = self._extract_authors(crossref_item, europepmc_record)

        return {
            "doi": str(doi),
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "journal_title": journal_title,
            "article_url": article_url or f"https://doi.org/{doi}",
            "pdf_url": pdf_url,
            "europepmc": europepmc_record,
        }

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        start_date, end_date = self._get_date_range()
        logger.info(f"Fetching subjournal papers from {start_date} to {end_date}")

        raw_papers: list[dict[str, Any]] = []
        seen_dois: set[str] = set()
        for journal_spec in self.journal_specs:
            logger.info(f"Querying Crossref for subjournal '{journal_spec.label}'")
            items = self._fetch_crossref_items(journal_spec, start_date, end_date)
            logger.info(f"Found {len(items)} Crossref matches for '{journal_spec.label}'")
            for item in items:
                doi = str(item.get("DOI") or "")
                if not doi or doi in seen_dois:
                    continue

                raw_entry = self._build_raw_entry(item)
                if raw_entry is None:
                    continue

                seen_dois.add(doi)
                raw_papers.append(raw_entry)

        if self.config.executor.debug:
            raw_papers = raw_papers[:10]
        return raw_papers

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        doi = raw_paper["doi"]
        return Paper(
            source=self.name,
            title=raw_paper["title"],
            authors=raw_paper["authors"],
            abstract=raw_paper["abstract"],
            url=f"https://doi.org/{doi}",
            pdf_url=raw_paper["pdf_url"],
            full_text=None,
            venue=raw_paper["journal_title"],
        )
