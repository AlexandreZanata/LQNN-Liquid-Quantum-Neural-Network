"""Smart search agent -- Wikipedia-first with quality-filtered fallback.

Search priority:
1. Wikipedia REST API (clean, curated, authoritative)
2. DuckDuckGo with domain whitelist/blacklist filtering
3. Multi-language support (search in both user language and English)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from urllib.parse import quote_plus

import aiohttp

from lqnn.agents.browser_agent import (
    BrowserAgent, SearchResult, _HEADERS, FETCH_TIMEOUT,
)

log = logging.getLogger(__name__)

WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
WIKIPEDIA_PT_SUMMARY_URL = "https://pt.wikipedia.org/api/rest_v1/page/summary/{}"
WIKIPEDIA_SEARCH_URL = (
    "https://en.wikipedia.org/w/api.php?action=query&list=search"
    "&srsearch={}&srlimit=3&format=json"
)


@dataclass
class SmartSearchResult:
    """Extended search result with source quality metadata."""
    query: str
    results: list[dict] = field(default_factory=list)
    success: bool = False
    source: str = ""
    wiki_summary: str = ""
    wiki_image_url: str = ""


class SmartSearchAgent:
    """Wikipedia-first intelligent search with quality filtering.

    Falls back to DuckDuckGo only when Wikipedia has no results.
    Prioritizes whitelisted domains and rejects blacklisted ones.
    """

    def __init__(self, browser: BrowserAgent | None = None) -> None:
        self.browser = browser or BrowserAgent()
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=_HEADERS, timeout=FETCH_TIMEOUT,
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def search(self, concept: str) -> SearchResult:
        """Search with Wikipedia priority, DuckDuckGo fallback.

        Returns a standard SearchResult for compatibility with AgentManager.
        """
        wiki_result = await self._search_wikipedia(concept)
        if wiki_result.success and wiki_result.results:
            return SearchResult(
                query=concept,
                results=wiki_result.results,
                success=True,
            )

        wiki_pt = await self._search_wikipedia_pt(concept)
        if wiki_pt.success and wiki_pt.results:
            return SearchResult(
                query=concept,
                results=wiki_pt.results,
                success=True,
            )

        ddg_result = await self.browser.search(concept)
        return ddg_result

    async def search_with_context(self, concept: str) -> SmartSearchResult:
        """Full search returning Wikipedia summary + structured results."""
        result = SmartSearchResult(query=concept)

        summary = await self._get_wiki_summary(concept)
        if summary:
            result.wiki_summary = summary.get("extract", "")
            result.wiki_image_url = summary.get("thumbnail", {}).get("source", "")
            result.results.append({
                "url": summary.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "title": summary.get("title", concept),
                "text": summary.get("extract", ""),
                "source": "wikipedia",
            })
            result.success = True
            result.source = "wikipedia"

        if not result.success:
            summary_pt = await self._get_wiki_summary_pt(concept)
            if summary_pt:
                result.wiki_summary = summary_pt.get("extract", "")
                result.wiki_image_url = summary_pt.get("thumbnail", {}).get("source", "")
                result.results.append({
                    "url": summary_pt.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "title": summary_pt.get("title", concept),
                    "text": summary_pt.get("extract", ""),
                    "source": "wikipedia_pt",
                })
                result.success = True
                result.source = "wikipedia_pt"

        if not result.success:
            ddg = await self.browser.search(concept)
            if ddg.success:
                result.results = ddg.results
                result.success = True
                result.source = "duckduckgo"

        return result

    async def fetch_wiki_image(self, concept: str) -> bytes | None:
        """Fetch the main Wikipedia image for a concept."""
        summary = await self._get_wiki_summary(concept)
        if not summary:
            summary = await self._get_wiki_summary_pt(concept)
        if not summary:
            return None

        img_url = summary.get("thumbnail", {}).get("source", "")
        if not img_url:
            img_url = summary.get("originalimage", {}).get("source", "")
        if not img_url:
            return None

        try:
            session = await self._ensure_session()
            async with session.get(img_url) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    if len(data) > 1000:
                        return data
        except Exception as e:
            log.debug("Failed to fetch wiki image for '%s': %s", concept, e)
        return None

    async def _get_wiki_summary(self, concept: str) -> dict | None:
        """Fetch Wikipedia EN summary via REST API."""
        return await self._fetch_wiki_summary(
            WIKIPEDIA_SUMMARY_URL.format(quote_plus(concept.replace(" ", "_")))
        )

    async def _get_wiki_summary_pt(self, concept: str) -> dict | None:
        """Fetch Wikipedia PT summary via REST API."""
        return await self._fetch_wiki_summary(
            WIKIPEDIA_PT_SUMMARY_URL.format(quote_plus(concept.replace(" ", "_")))
        )

    async def _fetch_wiki_summary(self, url: str) -> dict | None:
        try:
            session = await self._ensure_session()
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if data.get("type") == "standard" and data.get("extract"):
                    return data
                return None
        except Exception as e:
            log.debug("Wikipedia fetch failed: %s", e)
            return None

    async def _search_wikipedia(self, concept: str) -> SmartSearchResult:
        """Search Wikipedia and return page URLs."""
        result = SmartSearchResult(query=concept, source="wikipedia")

        summary = await self._get_wiki_summary(concept)
        if summary and summary.get("extract"):
            page_url = (
                summary.get("content_urls", {})
                .get("desktop", {})
                .get("page", "")
            )
            if page_url:
                result.results.append({
                    "url": page_url,
                    "title": summary.get("title", concept),
                })
                result.wiki_summary = summary.get("extract", "")
                result.success = True
                return result

        try:
            session = await self._ensure_session()
            url = WIKIPEDIA_SEARCH_URL.format(quote_plus(concept))
            async with session.get(url) as resp:
                if resp.status != 200:
                    return result
                data = await resp.json()

            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "")
                if title:
                    page_url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
                    result.results.append({
                        "url": page_url,
                        "title": title,
                    })
            result.success = bool(result.results)
        except Exception as e:
            log.debug("Wikipedia search failed for '%s': %s", concept, e)

        return result

    async def _search_wikipedia_pt(self, concept: str) -> SmartSearchResult:
        """Search Portuguese Wikipedia."""
        result = SmartSearchResult(query=concept, source="wikipedia_pt")

        summary = await self._get_wiki_summary_pt(concept)
        if summary and summary.get("extract"):
            page_url = (
                summary.get("content_urls", {})
                .get("desktop", {})
                .get("page", "")
            )
            if page_url:
                result.results.append({
                    "url": page_url,
                    "title": summary.get("title", concept),
                })
                result.wiki_summary = summary.get("extract", "")
                result.success = True

        return result
