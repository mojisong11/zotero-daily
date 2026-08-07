"""Tests for zotero_arxiv_daily.construct_email: render_email, get_stars, get_block_html."""

from zotero_arxiv_daily.construct_email import render_email, get_stars, get_block_html, get_empty_html
from tests.canned_responses import make_sample_paper


def test_render_email_with_papers():
    papers = [make_sample_paper(score=7.5, tldr="A great paper.", affiliations=["MIT"])]
    html = render_email({"arxiv": papers, "biorxiv": []})
    assert "Arxiv Papers" in html
    assert "BioRxiv Papers" in html
    assert "Sample Paper Title" in html
    assert "A great paper." in html
    assert "MIT" in html
    assert "arXiv ID" in html
    assert "2026.00001" in html


def test_render_email_empty_list():
    html = render_email({"arxiv": [], "biorxiv": []})
    assert "Arxiv Papers" in html
    assert "BioRxiv Papers" in html
    assert "No Papers Today" in html


def test_render_email_author_truncation():
    authors = [f"Author {i}" for i in range(10)]
    paper = make_sample_paper(authors=authors, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Author 0" in html
    assert "Author 1" in html
    assert "Author 2" in html
    assert "Author 3" in html
    assert "Author 4" in html
    assert "..." in html
    assert "Author 5" not in html
    assert "Author 9" not in html


def test_render_email_affiliation_truncation():
    affiliations = [f"Uni {i}" for i in range(8)]
    paper = make_sample_paper(affiliations=affiliations, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Uni 0" in html
    assert "Uni 4" in html
    assert "..." in html
    assert "Uni 7" not in html


def test_render_email_no_affiliations():
    paper = make_sample_paper(affiliations=None, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Unknown Affiliation" in html


def test_get_stars_low_score():
    assert get_stars(5.0) == ""
    assert get_stars(6.0) == ""


def test_get_stars_high_score():
    stars = get_stars(8.0)
    assert stars.count("full-star") == 5


def test_get_stars_mid_score():
    stars = get_stars(7.0)
    assert "star" in stars
    assert stars.count("full-star") + stars.count("half-star") > 0


def test_get_block_html_contains_all_fields():
    html = get_block_html("Title", "Auth", "3.5", "arXiv ID", "2026.00001", "Summary", "http://pdf.url", "MIT")
    assert "Title" in html
    assert "Auth" in html
    assert "3.5" in html
    assert "arXiv ID" in html
    assert "2026.00001" in html
    assert "Summary" in html
    assert "http://pdf.url" in html
    assert "MIT" in html


def test_render_email_extracts_biorxiv_doi():
    paper = make_sample_paper(
        source="biorxiv",
        url="https://www.biorxiv.org/content/10.1101/2026.03.01.000001v1.full.pdf",
        pdf_url="https://www.biorxiv.org/content/10.1101/2026.03.01.000001v1.full.pdf",
        score=7.2,
        tldr="Bio summary",
    )
    html = render_email({"biorxiv": [paper]})
    assert "bioRxiv DOI" in html
    assert "10.1101/2026.03.01.000001" in html


def test_render_email_renders_subjournal_venue_and_doi():
    paper = make_sample_paper(
        source="subjournal",
        url="https://doi.org/10.1038/s41592-026-03192-w",
        pdf_url="https://www.nature.com/articles/s41592-026-03192-w.pdf",
        venue="Nature Methods",
        score=7.3,
        tldr="Subjournal summary",
    )
    html = render_email({"subjournal": [paper]})
    assert "Subjournal Papers" in html
    assert "Nature Methods" in html
    assert "DOI" in html
    assert "10.1038/s41592-026-03192-w" in html


def test_get_empty_html():
    html = get_empty_html()
    assert "No Papers Today" in html
