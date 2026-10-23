import csv
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "data" / "sources.csv"
RAW_DIR = ROOT / "data" / "raw"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
}


def download(client: httpx.Client, url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    with tmp.open("rb") as f:
        if f.read(5) != b"%PDF-":
            tmp.unlink()
            raise ValueError("La respuesta no es un PDF")

    tmp.rename(dest)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0

    with (
        SOURCES.open(encoding="utf-8") as f,
        httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60) as client,
    ):
        for row in csv.DictReader(f):
            dest = RAW_DIR / f"{row['id']}.pdf"
            if dest.exists():
                print(f"= {row['id']} (ya existe)")
                continue

            try:
                download(client, row["url"], dest)
                print(f"+ {row['id']}")
            except (httpx.HTTPError, ValueError) as e:
                failures += 1
                print(f"x {row['id']}: {e}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
