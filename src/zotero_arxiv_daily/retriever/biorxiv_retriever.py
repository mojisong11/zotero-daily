from datetime import datetime

import requests
from .base import BaseRetriever, register_retriever
from ..protocol import Paper
from loguru import logger
from typing import Any
from time import sleep

@register_retriever("biorxiv")
class BiorxivRetriever(BaseRetriever):
    server = "biorxiv"

    def __init__(self, config):
        super().__init__(config)
        if self.retriever_config.category is None:
            raise ValueError(f"category must be specified for {self.name}")

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        api_url = f"https://api.biorxiv.org/details/{self.server}/2d"
        retry_num = 10
        delay_time = 10

        for i in range(retry_num):
            try:
                response = requests.get(api_url)
                response.raise_for_status()

                # 核心修复：
                # bioRxiv 即使 HTTP 请求成功，
                # 也可能返回空内容或非 JSON。
                # 因此 json() 也必须参与 retry。
                result = response.json()

                break

            except Exception as e:
                if i == retry_num - 1:
                    logger.error(
                        f"Failed to retrieve {self.server} papers "
                        f"after {retry_num} attempts: {str(e)}. "
                        f"Skip this source."
                    )
                    return []

                logger.warning(
                    f"Failed to retrieve papers: {str(e)}. "
                    f"Retry in {delay_time} seconds."
                )
                sleep(delay_time)

        collection = result['collection']

        if len(collection) == 0:
            logger.warning(
                f"No paper found. API Message: {result['messages']}"
            )
            return []

        dated_collection = [
            (
                datetime.strptime(c['date'], "%Y-%m-%d").date(),
                c
            )
            for c in collection
        ]

        latest_date = max(
            date for date, _ in dated_collection
        )

        collection = [
            c
            for date, c in dated_collection
            if date == latest_date
        ]

        categories = [
            c.lower()
            for c in self.retriever_config.category
        ]

        collection = [
            c
            for c in collection
            if c['category'] in categories
        ]

        if self.config.executor.debug:
            collection = collection[:10]

        return collection

    def convert_to_paper(
        self,
        raw_paper: dict[str, Any]
    ) -> Paper | None:

        title = raw_paper['title']

        authors = [
            a.strip()
            for a in raw_paper['authors'].split(';')
        ]

        abstract = raw_paper['abstract']

        article_url = (
            f"https://www.{self.server}.org/content/"
            f"{raw_paper['doi']}v{raw_paper['version']}"
        )

        full_text = None  # biorxiv forbids scraping its pdf

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=article_url,
            pdf_url=article_url,
            full_text=full_text
        )
