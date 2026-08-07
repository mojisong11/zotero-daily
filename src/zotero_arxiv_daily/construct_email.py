from collections.abc import Mapping
import math
import re

from .protocol import Paper


SOURCE_SECTION_TITLES = {
    "arxiv": "Arxiv Papers",
    "biorxiv": "BioRxiv Papers",
    "medrxiv": "MedRxiv Papers",
    "subjournal": "Subjournal Papers",
}

SOURCE_IDENTIFIER_LABELS = {
    "arxiv": "arXiv ID",
    "biorxiv": "bioRxiv DOI",
    "medrxiv": "medRxiv DOI",
    "subjournal": "DOI",
}

ARXIV_ID_PATTERN = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)")
DOI_PATTERN = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

__SECTIONS__

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template

def get_block_html(
    title:str,
    authors:str,
    rate:str,
    identifier_label:str | None,
    identifier_value:str | None,
    tldr:str,
    pdf_url:str,
    affiliations:str=None,
    code_url:str | None=None,
):
    code = (
        f'<a href="{code_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #5bc0de; padding: 8px 16px; border-radius: 4px; margin-left: 8px;">Code</a>'
        if code_url else ''
    )
    identifier_html = ""
    if identifier_label and identifier_value:
        identifier_html = f"""
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>{identifier_label}:</strong> {identifier_value}
        </td>
    </tr>
"""
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    {identifier_html}
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {tldr}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
            {code}
        </td>
    </tr>
</table>
"""
    return block_template.format(
        title=title,
        authors=authors,
        rate=rate,
        identifier_html=identifier_html,
        tldr=tldr,
        pdf_url=pdf_url,
        affiliations=affiliations,
        code=code,
    )

def get_stars(score:float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def _normalize_sections(
    papers_or_sections: Mapping[str, list[Paper]] | list[Paper],
    papers_biorxiv: list[Paper] | None = None,
    papers_medrxiv: list[Paper] | None = None,
) -> dict[str, list[Paper]]:
    if isinstance(papers_or_sections, Mapping):
        return {str(source): list(papers) for source, papers in papers_or_sections.items()}

    sections = {"arxiv": list(papers_or_sections)}
    if papers_biorxiv is not None:
        sections["biorxiv"] = list(papers_biorxiv)
    if papers_medrxiv is not None:
        sections["medrxiv"] = list(papers_medrxiv)
    return sections


def _format_authors(authors: list[str]) -> str:
    formatted_authors = authors[:5]
    author_text = ', '.join(formatted_authors)
    if len(authors) > 5:
        author_text += ', ...'
    return author_text


def _format_secondary_line(paper: Paper) -> str:
    affiliations = paper.affiliations
    if affiliations is not None:
        shown_affiliations = affiliations[:5]
        affiliation_text = ', '.join(shown_affiliations)
        if len(affiliations) > 5:
            affiliation_text += ', ...'
        return affiliation_text

    if paper.venue:
        return paper.venue

    return 'Unknown Affiliation'


def _extract_arxiv_id(url: str | None) -> str | None:
    if not url:
        return None

    match = ARXIV_ID_PATTERN.search(url)
    if match is None:
        return None

    identifier = match.group(1).removesuffix(".pdf")
    return re.sub(r"v\d+$", "", identifier)


def _extract_doi(url: str | None) -> str | None:
    if not url:
        return None

    match = DOI_PATTERN.search(url)
    if match is None:
        return None
    return match.group(1)


def _get_identifier(paper: Paper) -> tuple[str | None, str | None]:
    identifier_label = SOURCE_IDENTIFIER_LABELS.get(paper.source)
    if paper.source == "arxiv":
        identifier_value = _extract_arxiv_id(paper.url) or _extract_arxiv_id(paper.pdf_url)
    elif paper.source in {"biorxiv", "medrxiv", "subjournal"}:
        identifier_value = _extract_doi(paper.url) or _extract_doi(paper.pdf_url)
    else:
        identifier_value = None
    return identifier_label, identifier_value


def _render_section(section_title: str, papers: list[Paper]) -> str:
    parts = []
    if len(papers) == 0:
        content = get_empty_html()
        return f"<h1>{section_title}</h1>\n<div>\n    {content}\n</div>"

    for p in papers:
        rate = get_stars(p.score) if p.score is not None else 'Unknown'
        authors = _format_authors(p.authors)
        affiliations = _format_secondary_line(p)
        identifier_label, identifier_value = _get_identifier(p)
        parts.append(
            get_block_html(
                p.title,
                authors,
                rate,
                identifier_label,
                identifier_value,
                p.tldr,
                p.pdf_url,
                affiliations,
                getattr(p, "code_url", None),
            )
        )

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    return f"<h1>{section_title}</h1>\n<div>\n    {content}\n</div>"


def render_email(
    papers_or_sections: Mapping[str, list[Paper]] | list[Paper],
    papers_biorxiv: list[Paper] | None = None,
    papers_medrxiv: list[Paper] | None = None,
) -> str:
    sections = _normalize_sections(papers_or_sections, papers_biorxiv, papers_medrxiv)
    if len(sections) == 0:
        sections = {"arxiv": []}

    rendered_sections = []
    for source, papers in sections.items():
        section_title = SOURCE_SECTION_TITLES.get(source, f"{source.title()} Papers")
        rendered_sections.append(_render_section(section_title, papers))

    return framework.replace('__SECTIONS__', '\n\n'.join(rendered_sections))
