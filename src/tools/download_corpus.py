import argparse
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

USER_AGENT = "LocalRAGCorpusDownloader/1.0 (+public-docs; respectful-crawler)"
REQUEST_TIMEOUT = 30
RATE_LIMIT_SECONDS = 1.0
CHUNK_SIZE = 64 * 1024

DEFAULT_SEED_URLS: Dict[str, List[str]] = {
    "cnil": [
        "https://www.cnil.fr/sites/default/files/atoms/files/guide-de-la-securite-des-donnees-personnelles.pdf",
        "https://www.cnil.fr/sites/default/files/atoms/files/cnil_guide_teletravail_securite.pdf",
        "https://www.cnil.fr/sites/default/files/atoms/files/cnil-guide-violations-de-donnees-personnelles.pdf",
        "https://www.cnil.fr/sites/default/files/atoms/files/cnil-guide-aipd.pdf",
        "https://www.cnil.fr/sites/default/files/atoms/files/cnil-guide-sous-traitant.pdf",
    ],
    "data_gouv": [
        "https://www.data.gouv.fr/fr/datasets/rapport-annuel-dactivite-2022/",
        "https://www.data.gouv.fr/fr/datasets/rapport-annuel-dactivite-2021/",
        "https://www.data.gouv.fr/fr/datasets/rapports-dactivite/",
        "https://www.data.gouv.fr/fr/datasets/rapport-dactivite-de-la-cnil/",
        "https://www.data.gouv.fr/fr/datasets/rapports-annuels/",
    ],
}


@dataclass
class DownloadResult:
    url: str
    filename: str
    source: str
    date_download: str
    status: str
    sha256: Optional[str]
    content_type: Optional[str]
    size_bytes: Optional[int]


class RateLimitedSession:
    def __init__(self, rate_limit_seconds: float = RATE_LIMIT_SECONDS):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.rate_limit_seconds = rate_limit_seconds
        self._last_request_ts = 0.0

    def close(self) -> None:
        self.session.close()

    def _wait_for_slot(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((requests.RequestException,)),
    )
    def get(self, url: str, stream: bool = False) -> requests.Response:
        self._wait_for_slot()
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT, stream=stream, allow_redirects=True)
        finally:
            self._last_request_ts = time.time()

        if response.status_code >= 500:
            raise requests.HTTPError(f"Server error {response.status_code} for {url}", response=response)
        return response


class CorpusDownloader:
    def __init__(self, base_dir: Path, seed_urls: Dict[str, List[str]]):
        self.base_dir = base_dir
        self.seed_urls = seed_urls
        self.raw_dir = base_dir / "data" / "raw"
        self.manifest_path = base_dir / "data" / "manifest.jsonl"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        self.session = RateLimitedSession()
        self.robot_parsers: Dict[str, Optional[RobotFileParser]] = {}
        self.seen_sha256: Set[str] = self._load_existing_hashes()

    def _load_existing_hashes(self) -> Set[str]:
        hashes: Set[str] = set()
        if not self.manifest_path.exists():
            return hashes

        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sha = row.get("sha256")
                if sha:
                    hashes.add(sha)
        return hashes

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append_manifest(self, item: DownloadResult) -> None:
        payload = {
            "url": item.url,
            "filename": item.filename,
            "source": item.source,
            "date_download": item.date_download,
            "status": item.status,
            "sha256": item.sha256,
            "content_type": item.content_type,
            "size_bytes": item.size_bytes,
        }
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _safe_filename(self, url: str, source: str, content_type: Optional[str]) -> str:
        parsed = urlparse(url)
        basename = os.path.basename(parsed.path) or "document"
        basename = re.sub(r"[^A-Za-z0-9._-]", "_", basename)

        if "." not in basename:
            if content_type and "pdf" in content_type.lower():
                basename = f"{basename}.pdf"
            else:
                basename = f"{basename}.bin"

        source_dir = self.raw_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)

        candidate = basename
        stem = Path(basename).stem
        suffix = Path(basename).suffix
        i = 1
        while (source_dir / candidate).exists():
            candidate = f"{stem}_{i}{suffix}"
            i += 1
        return candidate

    def _get_robot_parser(self, url: str) -> Optional[RobotFileParser]:
        parsed = urlparse(url)
        host_key = f"{parsed.scheme}://{parsed.netloc}"
        if host_key in self.robot_parsers:
            return self.robot_parsers[host_key]

        robots_url = urljoin(host_key, "/robots.txt")
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
            self.robot_parsers[host_key] = rp
        except Exception:
            self.robot_parsers[host_key] = None
        return self.robot_parsers[host_key]

    def _is_allowed_by_robots(self, url: str) -> bool:
        rp = self._get_robot_parser(url)
        if rp is None:
            return True
        return rp.can_fetch(USER_AGENT, url)

    def _is_pdf_candidate(self, url: str, anchor_text: str = "") -> bool:
        lowered = url.lower()
        if lowered.endswith(".pdf"):
            return True
        if "format=pdf" in lowered or "type=pdf" in lowered:
            return True
        if "pdf" in anchor_text.lower():
            return True
        return False

    def _extract_data_gouv_pdf_links(self, dataset_url: str) -> List[str]:
        links: List[str] = []
        if not self._is_allowed_by_robots(dataset_url):
            print(f"[SKIP robots] {dataset_url}")
            return links

        try:
            resp = self.session.get(dataset_url)
        except Exception as exc:
            print(f"[WARN] Failed to fetch dataset page: {dataset_url} ({exc})")
            return links

        if resp.status_code != 200:
            print(f"[WARN] Dataset page returned {resp.status_code}: {dataset_url}")
            return links

        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href:
                continue
            full_url = urljoin(dataset_url, href)
            if self._is_pdf_candidate(full_url, a.get_text(" ", strip=True)):
                links.append(full_url)

        # Keep unique order
        seen: Set[str] = set()
        uniq_links: List[str] = []
        for link in links:
            if link not in seen:
                uniq_links.append(link)
                seen.add(link)
        return uniq_links

    def _download_file(self, url: str, source: str) -> DownloadResult:
        date_download = self._now_iso()

        if not self._is_allowed_by_robots(url):
            return DownloadResult(
                url=url,
                filename="",
                source=source,
                date_download=date_download,
                status="skipped_robots",
                sha256=None,
                content_type=None,
                size_bytes=None,
            )

        try:
            resp = self.session.get(url, stream=True)
        except Exception as exc:
            return DownloadResult(
                url=url,
                filename="",
                source=source,
                date_download=date_download,
                status=f"error_request:{type(exc).__name__}",
                sha256=None,
                content_type=None,
                size_bytes=None,
            )

        if resp.status_code == 404:
            return DownloadResult(
                url=url,
                filename="",
                source=source,
                date_download=date_download,
                status="error_404",
                sha256=None,
                content_type=resp.headers.get("Content-Type"),
                size_bytes=None,
            )

        if resp.status_code != 200:
            return DownloadResult(
                url=url,
                filename="",
                source=source,
                date_download=date_download,
                status=f"error_http_{resp.status_code}",
                sha256=None,
                content_type=resp.headers.get("Content-Type"),
                size_bytes=None,
            )

        content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower() or None
        filename = self._safe_filename(url, source, content_type)
        source_dir = self.raw_dir / source
        target_path = source_dir / filename
        temp_path = source_dir / f".{filename}.part"

        sha = hashlib.sha256()
        size = 0
        first_chunk = b""

        try:
            with temp_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    if not first_chunk:
                        first_chunk = chunk[:16]
                    sha.update(chunk)
                    size += len(chunk)
                    f.write(chunk)
        except Exception:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            return DownloadResult(
                url=url,
                filename="",
                source=source,
                date_download=date_download,
                status="error_stream",
                sha256=None,
                content_type=content_type,
                size_bytes=None,
            )

        is_probably_pdf = (content_type and "pdf" in content_type) or first_chunk.startswith(b"%PDF")
        if not is_probably_pdf:
            temp_path.unlink(missing_ok=True)
            return DownloadResult(
                url=url,
                filename="",
                source=source,
                date_download=date_download,
                status="skipped_non_pdf",
                sha256=None,
                content_type=content_type,
                size_bytes=size,
            )

        digest = sha.hexdigest()
        if digest in self.seen_sha256:
            temp_path.unlink(missing_ok=True)
            return DownloadResult(
                url=url,
                filename="",
                source=source,
                date_download=date_download,
                status="duplicate_sha256",
                sha256=digest,
                content_type=content_type,
                size_bytes=size,
            )

        shutil.move(str(temp_path), str(target_path))
        self.seen_sha256.add(digest)

        return DownloadResult(
            url=url,
            filename=str(target_path.relative_to(self.base_dir)),
            source=source,
            date_download=date_download,
            status="downloaded",
            sha256=digest,
            content_type=content_type,
            size_bytes=size,
        )

    def _build_download_plan(self) -> List[Tuple[str, str]]:
        plan: List[Tuple[str, str]] = []

        for source, urls in self.seed_urls.items():
            if source == "data_gouv":
                for dataset_url in urls:
                    pdf_links = self._extract_data_gouv_pdf_links(dataset_url)
                    if not pdf_links:
                        print(f"[INFO] No PDF links found on dataset page: {dataset_url}")
                        continue
                    for pdf_url in pdf_links:
                        plan.append((source, pdf_url))
                continue

            for url in urls:
                # For non-data_gouv sources, allow direct file endpoints even without ".pdf"
                # extension (examples: "/open", "/download", ".ashx").
                plan.append((source, url))

        # Deduplicate by URL while keeping order
        deduped: List[Tuple[str, str]] = []
        seen: Set[str] = set()
        for source, url in plan:
            if url not in seen:
                deduped.append((source, url))
                seen.add(url)
        return deduped

    def run(self) -> None:
        plan = self._build_download_plan()
        if not plan:
            print("[INFO] Nothing to download from seed URLs.")
            return

        print(f"[INFO] Planned downloads: {len(plan)}")

        for source, url in tqdm(plan, desc="Downloading", unit="file"):
            result = self._download_file(url, source=source)
            self._append_manifest(result)
            print(f"[{result.status}] {url}")

    def close(self) -> None:
        self.session.close()


def load_seed_urls(seed_file: Optional[Path]) -> Dict[str, List[str]]:
    if seed_file is None:
        return DEFAULT_SEED_URLS

    with seed_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Seed file must be a JSON object with source keys.")

    normalized: Dict[str, List[str]] = {}
    for source, urls in data.items():
        if not isinstance(source, str) or not source.strip():
            continue
        if not isinstance(urls, list):
            raise ValueError(f"Seed '{source}' must be a list of URLs.")
        normalized[source.strip()] = [u for u in urls if isinstance(u, str) and u.strip()]
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public PDF corpus for local RAG")
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help="Optional JSON file with seed URLs per source (special parsing for key: data_gouv)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root (defaults to repo root)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_urls = load_seed_urls(args.seed_file)

    downloader = CorpusDownloader(base_dir=args.base_dir, seed_urls=seed_urls)
    try:
        downloader.run()
    finally:
        downloader.close()


if __name__ == "__main__":
    main()
