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

        result = None

        for i in range(retry_num):
            try:
                response = requests.get(
                    api_url,
                    timeout=30,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 "
                            "(Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 "
                            "(KHTML, like Gecko) "
                            "Chrome/120.0 Safari/537.36"
                        )
                    },
                )

                # 检查 HTTP 状态，例如 403、429、500、502、503
                response.raise_for_status()

                # 防止服务器返回空响应
                if not response.text.strip():
                    raise ValueError(
                        f"{self.server} API returned an empty response"
                    )

                # 必须放在 retry 循环内部
                # 否则 HTTP 200 但返回 HTML/空内容时不会重试
                result = response.json()

                # 检查 API 返回结构
                if not isinstance(result, dict):
                    raise ValueError(
                        f"{self.server} API returned unexpected data type: "
                        f"{type(result)}"
                    )

                if "collection" not in result:
                    raise ValueError(
                        f"{self.server} API response does not contain "
                        f"'collection'. Response: {response.text[:500]}"
                    )

                # 请求及 JSON 解析均成功
                break

            except Exception as e:
                logger.warning(
                    f"Failed to retrieve {self.server} papers "
                    f"(attempt {i + 1}/{retry_num}): {str(e)}"
                )

                # 输出服务器实际返回的信息，方便以后排查
                if "response" in locals():
                    logger.warning(
                        f"{self.server} API response: "
                        f"status={response.status_code}, "
                        f"content-type={response.headers.get('Content-Type')}, "
                        f"body={response.text[:300]!r}"
                    )

                if i == retry_num - 1:
                    # 不让 bioRxiv 临时故障导致整个每日任务退出
                    logger.error(
                        f"Failed to retrieve {self.server} papers after "
                        f"{retry_num} attempts. Skip {self.server}."
                    )
                    return []

                logger.warning(
                    f"Retry in {delay_time} seconds."
                )
                sleep(delay_time)

        # 理论上不会走到这里，但作为额外保护
        if result is None:
            logger.error(
                f"No valid response received from {self.server}. "
                f"Skip {self.server}."
            )
            return []

        collection = result.get("collection", [])

        if len(collection) == 0:
            logger.warning(
                f"No paper found for {self.server}. "
                f"API Message: {result.get('messages', 'N/A')}"
            )
            return []

        # 获取日期并找最新一天
        dated_collection = []

        for c in collection:
            try:
                paper_date = datetime.strptime(
                    c["date"],
                    "%Y-%m-%d"
                ).date()

                dated_collection.append(
                    (paper_date, c)
                )

            except (KeyError, ValueError) as e:
                logger.warning(
                    f"Skip paper with invalid date: {str(e)}"
                )

        if len(dated_collection) == 0:
            logger.warning(
                f"No valid dated papers found for {self.server}."
            )
            return []

        latest_date = max(
            date for date, _ in dated_collection
        )

        collection = [
            c
            for date, c in dated_collection
            if date == latest_date
        ]

        # 配置 category 全部转为小写
        categories = [
            c.lower()
            for c in self.retriever_config.category
        ]

        # API 返回的 category 也转为小写再比较
        collection = [
            c
            for c in collection
            if c.get("category", "").lower() in categories
        ]

        if self.config.executor.debug:
            collection = collection[:10]

        return collection

    def convert_to_paper(
        self,
        raw_paper: dict[str, Any]
    ) -> Paper | None:

        title = raw_paper["title"]

        authors = [
            a.strip()
            for a in raw_paper["authors"].split(";")
        ]

        abstract = raw_paper["abstract"]

        article_url = (
            f"https://www.{self.server}.org/content/"
            f"{raw_paper['doi']}v{raw_paper['version']}"
        )

        # bioRxiv forbids scraping its PDF
        full_text = None

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=article_url,
            pdf_url=article_url,
            full_text=full_text,
        )
