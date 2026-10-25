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
KINDS = {"pdf", "html"}


def looks_valid(head: bytes, kind: str) -> bool:
    if kind == "pdf":
        return head.startswith(b"%PDF-")
    lowered = head.lower()
    return b"<html" in lowered or b"<!doctype html" in lowered


def download(client: httpx.Client, url: str, kind: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    with tmp.open("rb") as f:
        head = f.read(1024)
    if not looks_valid(head, kind):
        tmp.unlink()
        raise ValueError(f"la respuesta no es un {kind.upper()} válido")

    tmp.rename(dest)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0

    with (
        SOURCES.open(encoding="utf-8") as f,
        httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60) as client,
    ):
        for row in csv.DictReader(f):
            source_id, kind = row["id"], row["kind"]
            if kind not in KINDS:
                failures += 1
                print(f"x {source_id}: kind inválido '{kind}' (usa pdf o html)", file=sys.stderr)
                continue

            dest = RAW_DIR / f"{source_id}.{kind}"
            if dest.exists():
                print(f"= {source_id} (ya existe)")
                continue
            try:
                download(client, row["url"], kind, dest)
                print(f"+ {source_id}")
            except (httpx.HTTPError, ValueError) as e:
                failures += 1
                print(f"x {source_id}: {e}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
