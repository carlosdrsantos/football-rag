"""Download the Laws of the Game pages to data/raw/.

Kept separate from parsing so that reworking the chunking, which will happen
many times, never means hitting theifab.com again.
"""

import argparse
import time
from pathlib import Path

import httpx

from lotg.sources import LAW_PAGES

RAW_DIR = Path("data/raw/laws")
USER_AGENT = "lotg-rag/0.1 (personal learning project)"
DELAY_SECONDS = 1.0


def fetch_all(force: bool = False) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30.0
    ) as client:
        for page in LAW_PAGES:
            target = RAW_DIR / f"law-{page.number:02d}-{page.slug}.html"
            paths.append(target)

            if target.exists() and not force:
                print(f"skip   {target.name}")
                continue

            response = client.get(page.url)
            response.raise_for_status()
            target.write_text(response.text, encoding="utf-8")
            print(f"fetch  {target.name}  ({len(response.text):,} bytes)")
            time.sleep(DELAY_SECONDS)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Laws of the Game pages.")
    parser.add_argument("--force", action="store_true", help="re-download cached pages")
    args = parser.parse_args()

    paths = fetch_all(force=args.force)
    total = sum(p.stat().st_size for p in paths)
    print(f"\n{len(paths)} pages in {RAW_DIR}  ({total / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
