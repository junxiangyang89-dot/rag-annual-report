"""Download the exact annual reports listed in the tracked corpus manifest."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.download_reports import download_pdf


def main():
    data = Path(__file__).resolve().parents[1] / "data"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    raw = data / "raw_pdf"
    raw.mkdir(exist_ok=True)
    for entry in manifest:
        target = raw / entry["filename"]
        if target.is_file():
            continue
        with TemporaryDirectory(dir=raw) as scratch:
            part = Path(scratch) / target.name
            if download_pdf(entry["source_url"].replace("http://", "https://", 1), part):
                with part.open("rb") as stream:
                    valid = stream.read(5) == b"%PDF-"
                if valid:
                    part.replace(target)
                    continue
        raise SystemExit(f"下载失败，请按清单下载并保存为 {target.name}")


if __name__ == "__main__":
    main()
