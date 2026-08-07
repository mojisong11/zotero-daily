"""Tests for SubjournalRetriever."""

from types import SimpleNamespace

import pytest
from omegaconf import open_dict

from zotero_arxiv_daily.retriever.subjournal_retriever import SubjournalRetriever


SAMPLE_CROSSREF_NATURE_METHODS = {
    "status": "ok",
    "message": {
        "items": [
            {
                "DOI": "10.1038/s41592-026-03192-w",
                "type": "journal-article",
                "title": ["Inference of tumor spatial habitats"],
                "container-title": ["Nature Methods"],
                "author": [{"given": "Lin", "family": "Tang"}],
                "resource": {"primary": {"URL": "https://www.nature.com/articles/s41592-026-03192-w"}},
                "link": [{"URL": "https://www.nature.com/articles/s41592-026-03192-w.pdf", "content-type": "application/pdf"}],
                "abstract": "<jats:p>Crossref abstract.</jats:p>",
            },
            {
                "DOI": "10.1038/s41592-026-99999-x",
                "type": "journal-article",
                "title": ["Author Correction: Something happened"],
                "container-title": ["Nature Methods"],
            },
            {
                "DOI": "10.1038/example",
                "type": "journal-article",
                "title": ["Wrong journal article"],
                "container-title": ["Nature Reviews Methods Primers"],
            },
        ]
    },
}

SAMPLE_CROSSREF_PNAS = {
    "status": "ok",
    "message": {
        "items": [
            {
                "DOI": "10.1073/pnas.2417269122",
                "type": "journal-article",
                "title": ["Modulating the microbiome as an approach to anticancer drug development"],
                "container-title": ["Proceedings of the National Academy of Sciences"],
                "author": [{"given": "Ada", "family": "Lovelace"}],
                "resource": {"primary": {"URL": "https://www.pnas.org/doi/10.1073/pnas.2417269122"}},
                "link": [],
            }
        ]
    },
}

SAMPLE_EUROPEPMC = {
    "version": "6.9",
    "resultList": {
        "result": [
            {
                "source": "MED",
                "doi": "10.1038/s41592-026-03192-w",
                "abstractText": "Europe PMC abstract text.",
                "authorString": "Lin Tang",
            }
        ]
    },
}


def test_subjournal_retrieve_and_enrich(config, monkeypatch):
    import requests

    def _patched(url, **kwargs):
        if "query.container-title=Nature%20Methods" in url:
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: SAMPLE_CROSSREF_NATURE_METHODS)
        if "query.container-title=Proceedings%20of%20the%20National%20Academy%20of%20Sciences" in url:
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: SAMPLE_CROSSREF_PNAS)
        if "europepmc" in url and "10.1038/s41592-026-03192-w" in url:
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: SAMPLE_EUROPEPMC)
        if "europepmc" in url:
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {"resultList": {"result": []}})
        raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr(requests, "get", _patched)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.subjournal_retriever.sleep", lambda _: None)

    with open_dict(config.source):
        config.source.subjournal = {
            "journals": ["Nature Methods", "PNAS"],
            "days": 2,
            "use_europepmc": True,
            "rows_per_query": 10,
        }

    retriever = SubjournalRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == 2
    assert papers[0].source == "subjournal"
    assert papers[0].venue == "Nature Methods"
    assert papers[0].abstract == "Europe PMC abstract text."
    assert papers[0].authors == ["Lin Tang"]
    assert papers[1].venue == "Proceedings of the National Academy of Sciences"


def test_subjournal_falls_back_to_crossref_abstract(config, monkeypatch):
    import requests

    def _patched(url, **kwargs):
        if "query.container-title=Nature%20Methods" in url:
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: SAMPLE_CROSSREF_NATURE_METHODS)
        if "europepmc" in url:
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {"resultList": {"result": []}})
        raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr(requests, "get", _patched)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.subjournal_retriever.sleep", lambda _: None)

    with open_dict(config.source):
        config.source.subjournal = {
            "journals": ["Nature Methods"],
            "days": 2,
            "use_europepmc": True,
        }

    retriever = SubjournalRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == 1
    assert papers[0].abstract == "Crossref abstract."


def test_subjournal_requires_journals(config):
    with open_dict(config.source):
        config.source.subjournal = {"journals": []}
    with pytest.raises(ValueError, match="journals must be specified"):
        SubjournalRetriever(config)


def test_subjournal_date_range_can_exclude_today(config, monkeypatch):
    import zotero_arxiv_daily.retriever.subjournal_retriever as subjournal_module
    from datetime import datetime, timezone

    class FixedDateTime:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 8, 10, 9, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(subjournal_module, "datetime", FixedDateTime)

    with open_dict(config.source):
        config.source.subjournal = {
            "journals": ["Nature Methods"],
            "days": 7,
            "end_days_ago": 1,
            "use_europepmc": True,
        }

    retriever = SubjournalRetriever(config)
    start_date, end_date = retriever._get_date_range()

    assert start_date == "2026-08-03"
    assert end_date == "2026-08-09"
