"""Web crawler agent -- scrapes text and images for the learning pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
}

MAX_PAGE_BYTES = 2 * 1024 * 1024  # 2 MB
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
FETCH_TIMEOUT = aiohttp.ClientTimeout(total=15)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


@dataclass
class FetchResult:
    url: str
    text: str = ""
    images: list[bytes] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    success: bool = False
    error: str = ""


@dataclass
class SearchResult:
    query: str
    results: list[dict] = field(default_factory=list)
    success: bool = False


class BrowserAgent:
    """Fetches web pages and images for the knowledge pipeline.

    Uses aiohttp for async HTTP. No browser engine needed.
    """

    def __init__(self, max_images_per_page: int = 5,
                 max_pages_per_search: int = 5) -> None:
        self._max_images = max_images_per_page
        self._max_pages = max_pages_per_search
        self._session: aiohttp.ClientSession | None = None
        self._fetch_count = 0

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=_HEADERS, timeout=FETCH_TIMEOUT,
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def search(self, query: str) -> SearchResult:
        """Search DuckDuckGo HTML for results."""
        session = await self._ensure_session()
        url = "https://html.duckduckgo.com/html/"
        try:
            async with session.post(url, data={"q": query}) as resp:
                if resp.status != 200:
                    return SearchResult(query=query, success=False)
                html = await resp.text()

            soup = BeautifulSoup(html, "html.parser")
            results = []
            for a_tag in soup.select("a.result__a")[:self._max_pages]:
                href = a_tag.get("href", "")
                title = a_tag.get_text(strip=True)
                if href and title:
                    real_url = self._extract_ddg_url(href)
                    if real_url:
                        results.append({"url": real_url, "title": title})

            return SearchResult(query=query, results=results, success=bool(results))
        except Exception as exc:
            log.warning("Search failed for '%s': %s", query, exc)
            return SearchResult(query=query, success=False)

    async def fetch_page(self, url: str, download_images: bool = True) -> FetchResult:
        """Fetch a page: extract text, links, and optionally download images."""
        session = await self._ensure_session()
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return FetchResult(url=url, error=f"HTTP {resp.status}")
                html = await resp.read()
                if len(html) > MAX_PAGE_BYTES:
                    html = html[:MAX_PAGE_BYTES]

            soup = BeautifulSoup(html, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header",
                             "aside", "form", "button", "select", "noscript",
                             "iframe", "svg", "input", "textarea"]):
                tag.decompose()

            main_content = (
                soup.find("article")
                or soup.find("main")
                or soup.find(attrs={"role": "main"})
                or soup.find("div", class_=re.compile(r"content|article|post|entry", re.I))
                or soup
            )

            raw_text = main_content.get_text(separator="\n", strip=True)

            lines = []
            for line in raw_text.split("\n"):
                line = line.strip()
                if len(line) < 25:
                    continue
                alpha_count = sum(c.isalpha() or c.isspace() for c in line)
                if alpha_count / max(len(line), 1) < 0.6:
                    continue
                lines.append(line)
            text = "\n".join(lines)[:10000]

            links = []
            for a in soup.find_all("a", href=True):
                full = urljoin(url, a["href"])
                if full.startswith("http"):
                    links.append(full)

            image_urls = self._extract_image_urls(soup, url)
            images = []
            if download_images and image_urls:
                images = await self._download_images(
                    image_urls[:self._max_images], session,
                )

            self._fetch_count += 1
            return FetchResult(
                url=url, text=text, images=images,
                image_urls=image_urls, links=links[:20],
                success=True,
            )
        except Exception as exc:
            log.warning("Fetch failed for '%s': %s", url, exc)
            return FetchResult(url=url, error=str(exc))

    async def search_images(self, query: str, n: int = 5) -> list[bytes]:
        """Search for images and download them."""
        search_result = await self.search(f"{query} image")
        all_images: list[bytes] = []

        for r in search_result.results[:3]:
            page = await self.fetch_page(r["url"], download_images=True)
            all_images.extend(page.images)
            if len(all_images) >= n:
                break

        return all_images[:n]

    def _extract_image_urls(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        urls = []
        for img in soup.find_all("img", src=True):
            src = urljoin(base_url, img["src"])
            parsed = urlparse(src)
            ext = parsed.path.rsplit(".", 1)[-1].lower() if "." in parsed.path else ""
            if f".{ext}" in IMAGE_EXTENSIONS and len(src) < 500:
                urls.append(src)
        return urls[:self._max_images * 2]

    async def _download_images(self, urls: list[str],
                               session: aiohttp.ClientSession) -> list[bytes]:
        images = []
        for img_url in urls:
            try:
                async with session.get(img_url) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        if 1000 < len(data) < MAX_IMAGE_BYTES:
                            images.append(data)
            except Exception:
                continue
        return images

    @staticmethod
    def _extract_ddg_url(href: str) -> str | None:
        """Extract the real URL from a DuckDuckGo redirect link."""
        if "uddg=" in href:
            from urllib.parse import parse_qs, urlparse as up
            qs = parse_qs(up(href).query)
            real = qs.get("uddg", [""])[0]
            return real if real.startswith("http") else None
        if href.startswith("http"):
            return href
        return None

    def stats(self) -> dict:
        return {"fetch_count": self._fetch_count}
