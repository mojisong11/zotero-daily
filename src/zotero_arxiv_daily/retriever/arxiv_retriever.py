from .base import BaseRetriever, register_retriever

import multiprocessing
import os
from queue import Empty
from tempfile import TemporaryDirectory
from time import sleep
from typing import Any, Callable, TypeVar

import arxiv
import feedparser
import requests
from arxiv import Result as ArxivResult
from loguru import logger
from tqdm import tqdm

from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar


T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    """Download a file with a connection/read timeout."""
    with requests.get(
        url,
        stream=True,
        timeout=DOWNLOAD_TIMEOUT,
    ) as response:
        response.raise_for_status()

        with open(path, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(
            (
                "ok",
                func(*args),
            )
        )
    except Exception as exc:
        result_queue.put(
            (
                "error",
                f"{type(exc).__name__}: {exc}",
            )
        )


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()

    context = multiprocessing.get_context(
        "fork"
        if "fork" in start_methods
        else start_methods[0]
    )

    result_queue = context.Queue()

    process = context.Process(
        target=_run_in_subprocess,
        args=(
            result_queue,
            func,
            args,
        ),
    )

    process.start()

    try:
        status, payload = result_queue.get(
            timeout=timeout
        )

    except Empty:
        if process.is_alive():
            process.kill()

        process.join(5)

        result_queue.close()
        result_queue.join_thread()

        logger.warning(
            f"{operation} timed out for "
            f"{paper_title} after {timeout} seconds"
        )

        return None

    process.join(5)

    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(
        f"{operation} failed for "
        f"{paper_title}: {payload}"
    )

    return None


def _extract_text_from_pdf_worker(
    pdf_url: str,
) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(
            temp_dir,
            "paper.pdf",
        )

        _download_file(
            pdf_url,
            path,
        )

        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(
    html_url: str,
) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(
        html_url
    )

    if downloaded is None:
        raise ValueError(
            f"Failed to download HTML from {html_url}"
        )

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
    )

    if not text:
        raise ValueError(
            f"No text extracted from {html_url}"
        )

    return text


def _extract_text_from_tar_worker(
    source_url: str,
    paper_id: str,
    paper_title: str | None = None,
) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(
            temp_dir,
            "paper.tar.gz",
        )

        _download_file(
            source_url,
            path,
        )

        file_contents = extract_tex_code_from_tar(
            path,
            paper_id,
            paper_title=paper_title,
        )

        if (
            not file_contents
            or "all" not in file_contents
        ):
            raise ValueError(
                "Main tex file not found."
            )

        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):

    def __init__(self, config):
        super().__init__(config)

        if self.config.source.arxiv.category is None:
            raise ValueError(
                "category must be specified for arxiv."
            )

    def _retrieve_raw_papers(
        self,
    ) -> list[ArxivResult]:
        """
        Retrieve today's arXiv papers.

        Strategy
        --------
        1. Read paper IDs from the arXiv RSS feed.
        2. Query arXiv API in batches of 20.
        3. Retry HTTP 429 with exponential-ish backoff.
        4. If a batch request fails with 406 or another
           HTTP error, fall back to one-paper-at-a-time
           requests.
        5. If one individual paper still fails, skip
           only that paper rather than terminating the
           whole workflow.
        """

        client = arxiv.Client(
            num_retries=10,
            delay_seconds=10,
        )

        query = "+".join(
            self.config.source.arxiv.category
        )

        include_cross_list = (
            self.config.source.arxiv.get(
                "include_cross_list",
                False,
            )
        )

        # -------------------------------------------------
        # 1. Retrieve latest IDs from arXiv RSS
        # -------------------------------------------------

        feed = feedparser.parse(
            f"https://rss.arxiv.org/atom/{query}"
        )

        if (
            hasattr(feed, "feed")
            and "title" in feed.feed
            and "Feed error for query"
            in feed.feed.title
        ):
            raise Exception(
                f"Invalid ARXIV_QUERY: {query}."
            )

        allowed_announce_types = (
            {"new", "cross"}
            if include_cross_list
            else {"new"}
        )

        all_paper_ids = [
            entry.id.removeprefix(
                "oai:arXiv.org:"
            )
            for entry in feed.entries
            if entry.get(
                "arxiv_announce_type",
                "new",
            )
            in allowed_announce_types
        ]

        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        logger.info(
            f"Found {len(all_paper_ids)} "
            "arXiv papers from RSS feed"
        )

        raw_papers: list[ArxivResult] = []

        # -------------------------------------------------
        # 2. Retrieve complete metadata using arXiv API
        # -------------------------------------------------

        bar = tqdm(
            total=len(all_paper_ids)
        )

        batch_size = 20

        # Only mainly relevant to HTTP 429.
        max_batch_retries = 5
        batch_retry_delay = 30

        for start_index in range(
            0,
            len(all_paper_ids),
            batch_size,
        ):
            batch_ids = all_paper_ids[
                start_index:
                start_index + batch_size
            ]

            batch_number = (
                start_index // batch_size
            )

            search = arxiv.Search(
                id_list=batch_ids
            )

            batch_success = False

            # ---------------------------------------------
            # Try normal batch request
            # ---------------------------------------------

            for attempt in range(
                max_batch_retries
            ):
                try:
                    batch = list(
                        client.results(search)
                    )

                    raw_papers.extend(batch)

                    # We attempted all IDs in this batch.
                    bar.update(
                        len(batch_ids)
                    )

                    batch_success = True

                    break

                except arxiv.HTTPError as exc:

                    status = getattr(
                        exc,
                        "status",
                        None,
                    )

                    # -------------------------------------
                    # Rate limiting: HTTP 429
                    # -------------------------------------

                    if status == 429:
                        if (
                            attempt
                            < max_batch_retries - 1
                        ):
                            wait = (
                                batch_retry_delay
                                * (attempt + 1)
                            )

                            logger.warning(
                                "arXiv API HTTP 429 "
                                f"on batch {batch_number}. "
                                f"Retry "
                                f"{attempt + 1}/"
                                f"{max_batch_retries} "
                                f"in {wait}s."
                            )

                            sleep(wait)

                            continue

                        logger.warning(
                            "arXiv batch request "
                            f"{batch_number} still "
                            "returned HTTP 429 after "
                            f"{max_batch_retries} "
                            "attempts. Falling back "
                            "to individual requests."
                        )

                    # -------------------------------------
                    # 406 / 403 / 5xx / other HTTP errors
                    # -------------------------------------

                    else:
                        logger.warning(
                            "arXiv batch request "
                            f"{batch_number} failed "
                            f"with HTTP {status}. "
                            "Falling back to "
                            "individual paper requests."
                        )

                    # Exit the batch retry loop and
                    # perform the single-paper fallback.
                    break

                except Exception as exc:
                    logger.warning(
                        "Unexpected error while "
                        f"retrieving arXiv batch "
                        f"{batch_number}: "
                        f"{type(exc).__name__}: {exc}. "
                        "Falling back to individual "
                        "paper requests."
                    )

                    break

            # ---------------------------------------------
            # 3. Batch succeeded
            # ---------------------------------------------

            if batch_success:
                if (
                    start_index + batch_size
                    < len(all_paper_ids)
                ):
                    sleep(3)

                continue

            # ---------------------------------------------
            # 4. Batch failed:
            #    retrieve each paper individually
            # ---------------------------------------------

            logger.info(
                f"Retrieving batch {batch_number} "
                f"paper-by-paper "
                f"({len(batch_ids)} papers)"
            )

            recovered_count = 0
            failed_count = 0

            for index, paper_id in enumerate(
                batch_ids
            ):
                try:
                    single_search = arxiv.Search(
                        id_list=[paper_id]
                    )

                    single_results = list(
                        client.results(
                            single_search
                        )
                    )

                    if single_results:
                        raw_papers.extend(
                            single_results
                        )

                        recovered_count += len(
                            single_results
                        )

                    else:
                        failed_count += 1

                        logger.warning(
                            "No arXiv API result "
                            f"returned for {paper_id}"
                        )

                except arxiv.HTTPError as exc:
                    failed_count += 1

                    status = getattr(
                        exc,
                        "status",
                        None,
                    )

                    logger.warning(
                        "Skipping arXiv paper "
                        f"{paper_id}: "
                        f"HTTP {status}"
                    )

                except Exception as exc:
                    failed_count += 1

                    logger.warning(
                        "Skipping arXiv paper "
                        f"{paper_id}: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                finally:
                    # Count the paper as processed even
                    # when it was skipped.
                    bar.update(1)

                # Avoid hammering arXiv API.
                if (
                    index + 1
                    < len(batch_ids)
                ):
                    sleep(1)

            logger.info(
                f"Fallback for batch "
                f"{batch_number}: "
                f"recovered {recovered_count}, "
                f"failed {failed_count}"
            )

            # Pause between batches.
            if (
                start_index + batch_size
                < len(all_paper_ids)
            ):
                sleep(3)

        bar.close()

        logger.info(
            f"Successfully retrieved "
            f"{len(raw_papers)} arXiv papers"
        )

        return raw_papers

    def convert_to_paper(
        self,
        raw_paper: ArxivResult,
    ) -> Paper:

        title = raw_paper.title

        authors = [
            author.name
            for author in raw_paper.authors
        ]

        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url

        # Preferred extraction order:
        #
        # source tar -> HTML -> PDF

        full_text = extract_text_from_tar(
            raw_paper
        )

        if full_text is None:
            full_text = extract_text_from_html(
                raw_paper
            )

        if full_text is None:
            full_text = extract_text_from_pdf(
                raw_paper
            )

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(
    paper: ArxivResult,
) -> str | None:

    html_url = paper.entry_id.replace(
        "/abs/",
        "/html/",
    )

    try:
        return _extract_text_from_html_worker(
            html_url
        )

    except Exception as exc:
        logger.warning(
            "HTML extraction failed for "
            f"{paper.title}: {exc}"
        )

        return None


def extract_text_from_pdf(
    paper: ArxivResult,
) -> str | None:

    if paper.pdf_url is None:
        logger.warning(
            "No PDF URL available for "
            f"{paper.title}"
        )

        return None

    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(
    paper: ArxivResult,
) -> str | None:

    source_url = paper.source_url()

    if source_url is None:
        logger.warning(
            "No source URL available for "
            f"{paper.title}"
        )

        return None

    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (
            source_url,
            paper.entry_id,
            paper.title,
        ),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
