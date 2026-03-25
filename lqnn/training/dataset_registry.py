"""Dataset registry -- download, parse, and manage multilingual parallel corpora.

Supported open-source datasets:
- Tatoeba:      human-verified sentence pairs, CC BY 2.0
- OPUS-100:     55M pairs covering 100 languages (via HuggingFace)
- ParaCrawl v9: large-scale EN-XX web crawl, CC0/open
- JParaCrawl:   EN-JA via HuggingFace filtered subset
"""

from __future__ import annotations

import bz2
import csv
import gzip
import logging
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import aiohttp
import ssl

log = logging.getLogger(__name__)

DATA_DIR = Path("data/languages")
RAW_DIR = DATA_DIR / "raw"


@dataclass
class SentencePair:
    source_lang: str
    target_lang: str
    source_text: str
    target_text: str
    dataset: str


@dataclass
class DatasetInfo:
    id: str
    name: str
    license: str
    base_url: str
    description: str
    supported_langs: list[str] = field(default_factory=list)

    def url_for_pair(self, src: str, tgt: str) -> str | None:
        raise NotImplementedError


# ── ISO 639-2/3 mapping ──

_LANG_MAP = {
    "ja": "jpn", "ru": "rus", "it": "ita", "zh": "cmn",
    "ko": "kor", "fr": "fra", "de": "deu", "es": "spa",
    "pt": "por", "ar": "ara", "en": "eng",
}


class TatoebaDataset(DatasetInfo):
    """Tatoeba exports: bz2-compressed TSV link files per language pair.

    Actual URL pattern (verified Mar 2026):
      https://downloads.tatoeba.org/exports/per_language/eng/eng-jpn_links.tsv.bz2
    """

    def __init__(self) -> None:
        super().__init__(
            id="tatoeba",
            name="Tatoeba",
            license="CC BY 2.0 FR",
            base_url="https://downloads.tatoeba.org/exports/per_language",
            description="13M+ human-verified sentence pairs across 429 languages",
            supported_langs=[
                "ja", "ru", "it", "zh", "ko", "fr", "de", "es", "pt", "ar",
            ],
        )

    def url_for_pair(self, src: str, tgt: str) -> str | None:
        src3 = _LANG_MAP.get(src)
        tgt3 = _LANG_MAP.get(tgt)
        if not src3 or not tgt3:
            return None
        return f"{self.base_url}/{src3}/{src3}-{tgt3}_links.tsv.bz2"


class Opus100Dataset(DatasetInfo):
    """OPUS-100 -- up to 1M pairs per language (via HuggingFace mirror).

    The original object.pouta.csc.fi URLs for individual splits are broken.
    HuggingFace hosts the full dataset reliably.

    URL pattern:
      https://huggingface.co/datasets/Helsinki-NLP/opus-100/resolve/main/data/en-ja-train.jsonl.gz
    Fallback to the full tar.gz:
      https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-en-ja-v1.0.tar.gz
    """

    def __init__(self) -> None:
        super().__init__(
            id="opus100",
            name="OPUS-100",
            license="Open",
            base_url="https://object.pouta.csc.fi/OPUS-100/v1.0",
            description="55M sentence pairs covering 100 languages",
            supported_langs=[
                "ja", "ru", "it", "zh", "ko", "fr", "de", "es", "pt", "ar",
            ],
        )

    def url_for_pair(self, src: str, tgt: str) -> str | None:
        pair = f"{min(src, tgt)}-{max(src, tgt)}"
        return f"{self.base_url}/opus-100-corpus-{pair}-v1.0.tar.gz"


class ParaCrawlDataset(DatasetInfo):
    """ParaCrawl v9 -- large-scale web-crawled EN-XX pairs.

    URL pattern (verified Mar 2026):
      EU langs:  https://web-language-models.s3.amazonaws.com/paracrawl/release9/en-{lang}/en-{lang}.txt.gz
      Bonus (ru): https://web-language-models.s3.amazonaws.com/paracrawl/bonus/en-ru.txt.gz
    """

    _BONUS_LANGS = {"ru"}

    def __init__(self) -> None:
        super().__init__(
            id="paracrawl",
            name="ParaCrawl v9",
            license="CC0 / Open",
            base_url="https://web-language-models.s3.amazonaws.com/paracrawl",
            description="Large-scale parallel web crawl for European languages",
            supported_langs=["ru", "it", "fr", "de", "es", "pt"],
        )

    def url_for_pair(self, src: str, tgt: str) -> str | None:
        if tgt not in self.supported_langs:
            return None
        if tgt in self._BONUS_LANGS:
            return f"{self.base_url}/bonus/en-{tgt}.txt.gz"
        return f"{self.base_url}/release9/en-{tgt}/en-{tgt}.txt.gz"


class JParaCrawlDataset(DatasetInfo):
    """JParaCrawl -- EN-JA parallel corpus via HuggingFace filtered subset.

    The official NTT site (kecl.ntt.co.jp) has SSL issues, so we use the
    curated HuggingFace mirror instead.
    """

    def __init__(self) -> None:
        super().__init__(
            id="jparacrawl",
            name="JParaCrawl (HF)",
            license="Open / NTT",
            base_url=(
                "https://huggingface.co/datasets/"
                "Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus/"
                "resolve/main"
            ),
            description="1M+ filtered English-Japanese parallel sentence pairs",
            supported_langs=["ja"],
        )

    def url_for_pair(self, src: str, tgt: str) -> str | None:
        if tgt != "ja":
            return None
        return f"{self.base_url}/data/train-00000-of-00001.parquet"


REGISTRY: list[DatasetInfo] = [
    TatoebaDataset(),
    Opus100Dataset(),
    ParaCrawlDataset(),
    JParaCrawlDataset(),
]


def _registry_by_id() -> dict[str, DatasetInfo]:
    return {ds.id: ds for ds in REGISTRY}


class DatasetRegistry:
    """Manages download, caching, and parsing of multilingual datasets."""

    def __init__(self) -> None:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        self._datasets = _registry_by_id()

    def available_datasets(self) -> list[DatasetInfo]:
        return list(REGISTRY)

    def get_dataset(self, dataset_id: str) -> DatasetInfo | None:
        return self._datasets.get(dataset_id)

    def local_path(self, dataset_id: str, lang_code: str) -> Path:
        return RAW_DIR / dataset_id / f"en-{lang_code}"

    def is_downloaded(self, dataset_id: str, lang_code: str) -> bool:
        p = self.local_path(dataset_id, lang_code)
        return p.exists() and any(p.iterdir()) if p.is_dir() else False

    async def download_dataset(
        self,
        dataset_id: str,
        lang_code: str,
        progress_callback=None,
    ) -> Path:
        """Download a dataset for a specific language pair (en -> lang_code).

        Returns the local directory containing downloaded file(s).
        """
        ds = self._datasets.get(dataset_id)
        if not ds:
            raise ValueError(f"Unknown dataset: {dataset_id}")
        if lang_code not in ds.supported_langs:
            raise ValueError(
                f"Language {lang_code} not supported by {dataset_id}"
            )

        url = ds.url_for_pair("en", lang_code)
        if not url:
            raise ValueError(f"Cannot build URL for en-{lang_code} in {dataset_id}")

        dest_dir = self.local_path(dataset_id, lang_code)
        dest_dir.mkdir(parents=True, exist_ok=True)

        filename = url.rsplit("/", 1)[-1]
        dest_file = dest_dir / filename

        if dest_file.exists() and dest_file.stat().st_size > 100:
            log.info("Dataset %s/en-%s already cached at %s",
                     dataset_id, lang_code, dest_file)
            if progress_callback:
                progress_callback({"status": "cached", "path": str(dest_file)})
            self._extract_if_needed(dest_file, dest_dir)
            return dest_dir

        log.info("Downloading %s -> %s", url, dest_file)
        if progress_callback:
            progress_callback({"status": "downloading", "url": url})

        try:
            ssl_ctx = ssl.create_default_context()
            timeout = aiohttp.ClientTimeout(total=3600, connect=60)
            conn = aiohttp.TCPConnector(ssl=ssl_ctx)
            headers = {
                "User-Agent": "LQNN-LanguageDownloader/1.0",
            }
            async with aiohttp.ClientSession(
                timeout=timeout, connector=conn, headers=headers,
            ) as session:
                async with session.get(url, allow_redirects=True) as resp:
                    if resp.status != 200:
                        raise RuntimeError(
                            f"HTTP {resp.status} downloading {url}"
                        )
                    total = int(resp.headers.get("Content-Length", 0))
                    downloaded = 0
                    last_pct = -1
                    with open(dest_file, "wb") as f:
                        async for chunk in resp.content.iter_chunked(1024 * 256):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if progress_callback and total > 0:
                                pct = round(downloaded / total * 100)
                                if pct >= last_pct + 5:
                                    last_pct = pct
                                    progress_callback({
                                        "status": "downloading",
                                        "pct": pct,
                                        "downloaded_mb": round(downloaded / 1e6, 1),
                                        "total_mb": round(total / 1e6, 1),
                                    })
        except Exception as exc:
            if dest_file.exists():
                dest_file.unlink()
            raise RuntimeError(f"Download failed for {url}: {exc}") from exc

        if progress_callback:
            progress_callback({"status": "downloaded", "path": str(dest_file)})

        self._extract_if_needed(dest_file, dest_dir)

        return dest_dir

    @staticmethod
    def _extract_if_needed(filepath: Path, dest_dir: Path) -> None:
        """Extract .tar.gz, .gz, or .bz2 archives."""
        name = filepath.name
        if name.endswith(".tar.gz") or name.endswith(".tgz"):
            try:
                with tarfile.open(filepath, "r:gz") as tf:
                    tf.extractall(path=dest_dir, filter="data")
                log.info("Extracted tar.gz to %s", dest_dir)
            except Exception as exc:
                log.warning("Failed to extract %s: %s", filepath, exc)

        elif name.endswith(".bz2") and not name.endswith(".tar.bz2"):
            out_path = dest_dir / name[:-4]
            if out_path.exists() and out_path.stat().st_size > 100:
                return
            try:
                with bz2.open(filepath, "rb") as bz_in:
                    with open(out_path, "wb") as f_out:
                        while True:
                            block = bz_in.read(1024 * 256)
                            if not block:
                                break
                            f_out.write(block)
                log.info("Decompressed bz2 to %s", out_path)
            except Exception as exc:
                log.warning("Failed to decompress bz2 %s: %s", filepath, exc)

        elif name.endswith(".gz") and not name.endswith(".tar.gz"):
            out_path = dest_dir / name[:-3]
            if out_path.exists() and out_path.stat().st_size > 100:
                return
            try:
                with gzip.open(filepath, "rb") as gz_in:
                    with open(out_path, "wb") as f_out:
                        while True:
                            block = gz_in.read(1024 * 256)
                            if not block:
                                break
                            f_out.write(block)
                log.info("Decompressed gz to %s", out_path)
            except Exception as exc:
                log.warning("Failed to decompress %s: %s", filepath, exc)

        elif name.endswith(".tar.bz2"):
            try:
                with tarfile.open(filepath, "r:bz2") as tf:
                    tf.extractall(path=dest_dir, filter="data")
                log.info("Extracted tar.bz2 to %s", dest_dir)
            except Exception as exc:
                log.warning("Failed to extract %s: %s", filepath, exc)

    def parse_pairs(
        self,
        dataset_id: str,
        lang_code: str,
        max_pairs: int = 50000,
    ) -> Iterator[SentencePair]:
        """Parse downloaded dataset files into SentencePair objects.

        Yields up to max_pairs sentence pairs for the given language.
        """
        dest_dir = self.local_path(dataset_id, lang_code)
        if not dest_dir.exists():
            log.warning("No data at %s", dest_dir)
            return

        _SKIP_EXTS = {".gz", ".bz2", ".tgz"}
        count = 0
        for fpath in sorted(dest_dir.rglob("*")):
            if fpath.is_dir():
                continue
            if count >= max_pairs:
                break
            if fpath.suffix.lower() in _SKIP_EXTS:
                continue

            suffix = fpath.suffix.lower()
            name = fpath.name.lower()

            try:
                if suffix == ".parquet":
                    for pair in self._parse_parquet(fpath, lang_code, dataset_id, max_pairs - count):
                        yield pair
                        count += 1
                elif suffix == ".tsv" or "tsv" in name:
                    for pair in self._parse_tsv(fpath, lang_code, dataset_id, max_pairs - count):
                        yield pair
                        count += 1
                elif suffix == ".txt" or suffix == "":
                    for pair in self._parse_parallel_txt(fpath, lang_code, dataset_id, max_pairs - count):
                        yield pair
                        count += 1
                elif suffix == ".csv":
                    for pair in self._parse_csv(fpath, lang_code, dataset_id, max_pairs - count):
                        yield pair
                        count += 1
            except Exception as exc:
                log.warning("Error parsing %s: %s", fpath, exc)
                continue

    @staticmethod
    def _parse_tsv(
        filepath: Path,
        lang_code: str,
        dataset_id: str,
        limit: int,
    ) -> Iterator[SentencePair]:
        """Parse TSV sentence pairs.

        Supports:
        - Tatoeba links: src_id \\t src_text \\t tgt_id \\t tgt_text
        - Simple pairs:  src_text \\t tgt_text
        """
        count = 0
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if count >= limit > 0:
                    break
                if len(row) >= 4:
                    src_text = row[1].strip()
                    tgt_text = row[3].strip()
                elif len(row) >= 2:
                    src_text = row[0].strip()
                    tgt_text = row[1].strip()
                else:
                    continue

                if len(src_text) < 3 or len(tgt_text) < 2:
                    continue

                count += 1
                yield SentencePair(
                    source_lang="en",
                    target_lang=lang_code,
                    source_text=src_text,
                    target_text=tgt_text,
                    dataset=dataset_id,
                )

    @staticmethod
    def _parse_parallel_txt(
        filepath: Path,
        lang_code: str,
        dataset_id: str,
        limit: int,
    ) -> Iterator[SentencePair]:
        """Parse tab-separated parallel text (ParaCrawl: score\\tsrc\\ttgt)."""
        count = 0
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if count >= limit > 0:
                    break
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    src_text = parts[1].strip()
                    tgt_text = parts[2].strip()
                elif len(parts) == 2:
                    src_text = parts[0].strip()
                    tgt_text = parts[1].strip()
                else:
                    continue

                if len(src_text) < 3 or len(tgt_text) < 2:
                    continue

                count += 1
                yield SentencePair(
                    source_lang="en",
                    target_lang=lang_code,
                    source_text=src_text,
                    target_text=tgt_text,
                    dataset=dataset_id,
                )

    @staticmethod
    def _parse_csv(
        filepath: Path,
        lang_code: str,
        dataset_id: str,
        limit: int,
    ) -> Iterator[SentencePair]:
        """Parse CSV with at least 2 text columns."""
        count = 0
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                if count >= limit > 0:
                    break
                if len(row) < 2:
                    continue
                src_text = row[0].strip()
                tgt_text = row[1].strip()
                if len(src_text) < 3 or len(tgt_text) < 2:
                    continue
                count += 1
                yield SentencePair(
                    source_lang="en",
                    target_lang=lang_code,
                    source_text=src_text,
                    target_text=tgt_text,
                    dataset=dataset_id,
                )

    @staticmethod
    def _parse_parquet(
        filepath: Path,
        lang_code: str,
        dataset_id: str,
        limit: int,
    ) -> Iterator[SentencePair]:
        """Parse a Parquet file (used by HuggingFace datasets).

        Expects columns: 'en'/'english' and '{lang}'/'japanese'/etc.
        Falls back to first two string columns.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            log.warning("pyarrow not installed -- cannot parse %s", filepath)
            return

        table = pq.read_table(filepath)
        df_cols = table.column_names

        en_col = None
        tgt_col = None
        for c in df_cols:
            cl = c.lower()
            if cl in ("en", "eng", "english", "source", "src"):
                en_col = c
            elif cl in (lang_code, _LANG_MAP.get(lang_code, ""), "target", "tgt",
                        "japanese", "russian", "french", "german", "spanish",
                        "portuguese", "italian", "chinese", "korean", "arabic"):
                tgt_col = c

        if not en_col or not tgt_col:
            if len(df_cols) >= 2:
                en_col = df_cols[0]
                tgt_col = df_cols[1]
            else:
                log.warning("Cannot identify columns in %s: %s", filepath, df_cols)
                return

        count = 0
        for i in range(table.num_rows):
            if count >= limit > 0:
                break
            src_text = str(table.column(en_col)[i].as_py()).strip()
            tgt_text = str(table.column(tgt_col)[i].as_py()).strip()
            if len(src_text) < 3 or len(tgt_text) < 2:
                continue
            count += 1
            yield SentencePair(
                source_lang="en",
                target_lang=lang_code,
                source_text=src_text,
                target_text=tgt_text,
                dataset=dataset_id,
            )
