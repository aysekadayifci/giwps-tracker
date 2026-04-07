#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:30:31 2026

@author: aishakadayifci-orellana
"""

import io
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup


# --------------------------------------------------
# App config
# --------------------------------------------------
st.set_page_config(
    page_title="GIWPS Conflict Tracker",
    page_icon="🌍",
    layout="wide",
)

USER_AGENT = "GIWPS-Conflict-Tracker/Streamlit-1.0"
REQUEST_TIMEOUT = 25
DEFAULT_LIMIT = 60

RSS_SOURCES = [
    {"name": "ReliefWeb Updates", "url": "https://reliefweb.int/updates/rss.xml", "type": "report"},
    {"name": "UN News", "url": "https://news.un.org/feed/subscribe/en/news/all/rss.xml", "type": "news"},
    {"name": "UN Women", "url": "https://www.unwomen.org/en/news-stories/rss", "type": "report"},
    {"name": "UNICEF", "url": "https://www.unicef.org/rss.xml", "type": "report"},
    {"name": "International Crisis Group", "url": "https://www.crisisgroup.org/rss", "type": "report"},
    {"name": "Human Rights Watch", "url": "https://www.hrw.org/rss/news", "type": "report"},
]

HTML_SEARCH_SOURCES = [
    {
        "name": "ReliefWeb Search",
        "type": "report",
        "search_url": "https://reliefweb.int/updates?search={query}",
        "result_selector": "article",
        "title_selectors": ["h3 a", "h2 a", "a"],
        "snippet_selectors": ["p"],
        "date_selectors": ["time"],
    },
]

TAG_KEYWORDS: Dict[str, List[str]] = {
    "women": ["women", "woman", "female", "widows", "mothers"],
    "girls": ["girls", "girl", "adolescent girls"],
    "boys_children": ["boys", "boy", "children", "child", "youth", "adolescents", "infants"],
    "gbv": ["gender-based violence", "gender based violence", "gbv", "domestic violence", "intimate partner violence"],
    "sexual_violence": ["sexual violence", "rape", "crsv", "conflict-related sexual violence", "sexual exploitation", "sexual abuse"],
    "wps": ["women peace and security", "wps", "unscr 1325", "resolution 1325", "1325 agenda"],
    "nap": ["national action plan", "nap"],
    "conflict_impact": ["displacement", "internally displaced", "refugees", "humanitarian", "civilian", "ceasefire", "armed conflict", "shelling", "airstrike", "war", "violence"],
    "environment": ["environment", "climate", "water insecurity", "resource conflict", "drought", "food insecurity", "flood", "land degradation"],
}


# --------------------------------------------------
# Data model
# --------------------------------------------------
@dataclass
class SourceItem:
    country: str
    title: str
    url: str
    source: str
    published_at: str
    snippet: str
    source_type: str
    score: int
    matched_tags: str


# --------------------------------------------------
# Helpers
# --------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_url(url: str) -> str:
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.text


def normalize_date(date_str: str) -> str:
    if not date_str:
        return ""

    try:
        dt = parsedate_to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        pass

    common_formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d %B %Y",
        "%b %d, %Y",
    ]

    for fmt in common_formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            continue

    return date_str


def date_in_range(date_value: str, date_from: str, date_to: str) -> bool:
    if not date_value:
        return True

    try:
        normalized = normalize_date(date_value)
        dt = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        start = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(date_to, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        return start <= dt <= end
    except Exception:
        return True


def build_keyword_list(extra_terms: str = "") -> List[str]:
    words = [
        "women", "girls", "boys", "children", "gender-based violence", "sexual violence",
        "women peace and security", "wps", "national action plan", "nap",
        "environment", "climate", "impact of conflict", "conflict", "displacement"
    ]
    if extra_terms.strip():
        extras = [term.strip() for term in extra_terms.split(",") if term.strip()]
        words.extend(extras)
    return words


def score_text(country: str, text: str) -> tuple[int, List[str]]:
    lowered = text.lower()
    score = 0
    matched = []

    if country.lower() in lowered:
        score += 3

    for tag, words in TAG_KEYWORDS.items():
        hits = sum(1 for w in words if w.lower() in lowered)
        if hits > 0:
            matched.append(tag)
            score += min(hits, 3)

    strong_terms = [
        "women peace and security",
        "gender-based violence",
        "sexual violence",
        "national action plan",
        "resolution 1325",
        "conflict-related sexual violence",
    ]
    for term in strong_terms:
        if term in lowered:
            score += 2

    return score, matched


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return " ".join(text.split())


def parse_rss_items(xml_text: str) -> List[dict]:
    items = []
    root = ET.fromstring(xml_text)

    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or item.findtext("date") or "").strip()
        description = (item.findtext("description") or item.findtext("summary") or "").strip()
        items.append(
            {
                "title": title,
                "link": link,
                "date": pub_date,
                "description": BeautifulSoup(description, "html.parser").get_text(" ", strip=True),
            }
        )

    atom_entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
    for entry in atom_entries:
        title = (entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
        updated = (entry.findtext("{http://www.w3.org/2005/Atom}updated") or "").strip()
        summary = (entry.findtext("{http://www.w3.org/2005/Atom}summary") or "").strip()
        link = ""
        for link_elem in entry.findall("{http://www.w3.org/2005/Atom}link"):
            href = link_elem.attrib.get("href")
            if href:
                link = href
                break
        items.append(
            {
                "title": title,
                "link": link,
                "date": updated,
                "description": BeautifulSoup(summary, "html.parser").get_text(" ", strip=True),
            }
        )

    return items


# --------------------------------------------------
# Retrieval
# --------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_from_rss(country: str, date_from: str, date_to: str, extra_terms: str, limit: int) -> List[SourceItem]:
    keywords = build_keyword_list(extra_terms)
    collected: List[SourceItem] = []

    for source in RSS_SOURCES:
        try:
            xml_text = fetch_url(source["url"])
            feed_items = parse_rss_items(xml_text)
        except Exception:
            continue

        source_count = 0
        for entry in feed_items:
            title = entry["title"]
            link = entry["link"]
            published = normalize_date(entry["date"])
            snippet = entry["description"]

            combined = f"{title} {snippet}"
            lowered = combined.lower()

            has_country = country.lower() in lowered
            has_keyword = any(word.lower() in lowered for word in keywords)

            if not (has_country and has_keyword):
                continue
            if not date_in_range(published, date_from, date_to):
                continue

            score, matched = score_text(country, combined)
            collected.append(
                SourceItem(
                    country=country,
                    title=title,
                    url=link,
                    source=source["name"],
                    published_at=published,
                    snippet=snippet[:450],
                    source_type=source["type"],
                    score=score,
                    matched_tags=", ".join(matched),
                )
            )
            source_count += 1
            if source_count >= limit:
                break

    return collected


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_from_html_search(country: str, date_from: str, date_to: str, extra_terms: str, limit: int) -> List[SourceItem]:
    collected: List[SourceItem] = []
    query = f'{country} women girls boys children WPS GBV "sexual violence"'
    if extra_terms.strip():
        query += f" {extra_terms}"

    for source in HTML_SEARCH_SOURCES:
        try:
            url = source["search_url"].format(query=quote(query))
            html = fetch_url(url)
            soup = BeautifulSoup(html, "html.parser")
            blocks = soup.select(source["result_selector"])
        except Exception:
            continue

        source_count = 0
        for block in blocks:
            title = ""
            link = ""
            snippet = ""
            published = ""

            for selector in source["title_selectors"]:
                elem = block.select_one(selector)
                if elem:
                    title = elem.get_text(" ", strip=True)
                    link = elem.get("href", "")
                    break

            for selector in source["snippet_selectors"]:
                elem = block.select_one(selector)
                if elem:
                    snippet = elem.get_text(" ", strip=True)
                    break

            for selector in source["date_selectors"]:
                elem = block.select_one(selector)
                if elem:
                    published = elem.get("datetime", "") or elem.get_text(" ", strip=True)
                    break

            if not title:
                continue

            combined = f"{title} {snippet}"
            if country.lower() not in combined.lower():
                continue
            if not date_in_range(published, date_from, date_to):
                continue

            score, matched = score_text(country, combined)
            if score < 4:
                continue

            collected.append(
                SourceItem(
                    country=country,
                    title=title,
                    url=link,
                    source=source["name"],
                    published_at=normalize_date(published),
                    snippet=snippet[:450],
                    source_type=source["type"],
                    score=score,
                    matched_tags=", ".join(matched),
                )
            )
            source_count += 1
            if source_count >= limit:
                break

    return collected


@st.cache_data(show_spinner=False, ttl=3600)
def enrich_results_with_article_text(items_dicts: List[dict], country: str, date_from: str, date_to: str, limit: int) -> List[SourceItem]:
    items = [SourceItem(**item) for item in items_dicts]
    enriched: List[SourceItem] = []

    for item in items[:limit]:
        try:
            if not item.url.startswith("http"):
                enriched.append(item)
                continue

            html = fetch_url(item.url)
            text = extract_text_from_html(html)
            if country.lower() not in text.lower():
                enriched.append(item)
                continue
            if not date_in_range(item.published_at, date_from, date_to):
                continue

            score, matched = score_text(country, f"{item.title} {item.snippet} {text[:5000]}")
            item.score = score
            item.matched_tags = ", ".join(matched)
            if not item.snippet or len(item.snippet) < 60:
                item.snippet = text[:450]
            enriched.append(item)
        except Exception:
            enriched.append(item)

    enriched.extend(items[limit:])
    return enriched


def dedupe_items(items: List[SourceItem]) -> List[SourceItem]:
    seen = set()
    cleaned = []
    for item in items:
        key = (item.title.strip().lower(), item.url.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


def items_to_dataframe(items: List[SourceItem]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=[
            "country", "title", "url", "source", "published_at", "snippet", "source_type", "score", "matched_tags"
        ])
    return pd.DataFrame([asdict(item) for item in items])


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_markdown_report(country: str, date_from: str, date_to: str, df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"# GIWPS Conflict Tracker Report: {country}")
    lines.append("")
    lines.append(f"**Date range:** {date_from} to {date_to}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total results:** {len(df)}")
    lines.append("")

    if not df.empty:
        tag_counts = {}
        for tag_string in df["matched_tags"].fillna(""):
            for tag in [t.strip() for t in tag_string.split(",") if t.strip()]:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        if tag_counts:
            lines.append("## Top themes")
            lines.append("")
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                lines.append(f"- {tag}: {count}")
            lines.append("")

        lines.append("## Top sources")
        lines.append("")
        for source, count in df["source"].value_counts().head(10).items():
            lines.append(f"- {source}: {count}")
        lines.append("")

        lines.append("## Results")
        lines.append("")
        for _, row in df.iterrows():
            lines.append(f"### {row['title']}")
            lines.append(f"- Source: {row['source']}")
            lines.append(f"- Date: {row['published_at']}")
            lines.append(f"- Type: {row['source_type']}")
            lines.append(f"- Score: {row['score']}")
            lines.append(f"- Tags: {row['matched_tags']}")
            lines.append(f"- URL: {row['url']}")
            lines.append(f"- Snippet: {row['snippet']}")
            lines.append("")
    else:
        lines.append("No matching results were found for the selected inputs.")

    return "\n".join(lines)


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🌍 GIWPS Conflict Tracker")
st.caption(
    "No API keys required. This Streamlit app searches curated public sources, ranks results, and lets you download the output."
)

with st.sidebar:
    st.header("Search settings")
    country = st.text_input("Country", value="Sudan")

    today = datetime.now(timezone.utc).date()
    month_ago = today - timedelta(days=30)

    date_from = st.date_input("From date", value=month_ago)
    date_to = st.date_input("To date", value=today)
    limit = st.number_input("Max results", min_value=5, max_value=200, value=DEFAULT_LIMIT, step=5)
    extra_terms = st.text_input("Extra terms (comma-separated)", value="")

    st.markdown("### Sources")
    use_rss = st.checkbox("RSS feeds", value=True)
    use_html = st.checkbox("Site search pages", value=True)
    use_enrich = st.checkbox("Open top results and rescore", value=True)

    run_search = st.button("Run research", type="primary", use_container_width=True)

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if "report_text" not in st.session_state:
    st.session_state.report_text = ""

if run_search:
    if not country.strip():
        st.error("Please enter a country name.")
    elif date_from > date_to:
        st.error("The start date must be on or before the end date.")
    else:
        all_items: List[SourceItem] = []
        date_from_str = date_from.isoformat()
        date_to_str = date_to.isoformat()

        progress = st.progress(0)
        status = st.empty()
        step = 0
        total_steps = int(use_rss) + int(use_html) + int(use_enrich)
        total_steps = max(total_steps, 1)

        if use_rss:
            status.info("Searching RSS feeds...")
            rss_items = fetch_from_rss(country.strip(), date_from_str, date_to_str, extra_terms, int(limit))
            all_items.extend(rss_items)
            step += 1
            progress.progress(step / total_steps)

        if use_html:
            status.info("Searching site search pages...")
            html_items = fetch_from_html_search(country.strip(), date_from_str, date_to_str, extra_terms, int(limit))
            all_items.extend(html_items)
            step += 1
            progress.progress(step / total_steps)

        cleaned = dedupe_items(all_items)

        if use_enrich and cleaned:
            status.info("Opening top links and rescoring article text...")
            cleaned = enrich_results_with_article_text(
                [asdict(item) for item in cleaned],
                country.strip(),
                date_from_str,
                date_to_str,
                min(20, len(cleaned)),
            )
            step += 1
            progress.progress(step / total_steps)

        cleaned = [item for item in cleaned if item.score >= 4]
        cleaned.sort(key=lambda x: (x.score, x.published_at), reverse=True)
        cleaned = cleaned[: int(limit)]

        results_df = items_to_dataframe(cleaned)
        st.session_state.results_df = results_df
        st.session_state.report_text = build_markdown_report(country.strip(), date_from_str, date_to_str, results_df)

        progress.progress(1.0)
        status.success(f"Done. {len(results_df)} results ready.")

results_df = st.session_state.results_df
report_text = st.session_state.report_text

col1, col2, col3 = st.columns([1, 1, 1])
col1.metric("Results", len(results_df))
col2.metric("Unique sources", 0 if results_df.empty else results_df["source"].nunique())
col3.metric("Average score", "0.0" if results_df.empty else f"{results_df['score'].mean():.1f}")

if not results_df.empty:
    st.subheader("Results")
    display_df = results_df.copy()
    display_df = display_df[["published_at", "source_type", "source", "score", "matched_tags", "title", "url", "snippet"]]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("URL"),
            "snippet": st.column_config.TextColumn("Snippet", width="large"),
            "title": st.column_config.TextColumn("Title", width="large"),
        },
    )

    with st.expander("Top theme counts", expanded=False):
        tag_counts = {}
        for tag_string in results_df["matched_tags"].fillna(""):
            for tag in [t.strip() for t in tag_string.split(",") if t.strip()]:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if tag_counts:
            tag_df = pd.DataFrame(
                [{"tag": tag, "count": count} for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)]
            )
            st.dataframe(tag_df, use_container_width=True, hide_index=True)
        else:
            st.write("No tag counts available.")

    csv_bytes = convert_df_to_csv(results_df)
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"{country.strip().replace(' ', '_')}_giwps_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        label="Download Markdown report",
        data=report_text.encode("utf-8"),
        file_name=f"{country.strip().replace(' ', '_')}_giwps_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.subheader("Generated report preview")
    st.text_area("Report", value=report_text, height=300)
else:
    st.info("Run a search to see results here.")

with st.expander("How to run this app", expanded=False):
    st.code(
        "pip install streamlit pandas requests beautifulsoup4\n"
        "streamlit run giwps_conflict_tracker_streamlit.py",
        language="bash",
    )
