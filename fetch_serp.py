#!/usr/bin/env python
"""
fetch_serp.py  –  SerpAPI Google-Images downloader
import fetch_serp  ;  paths = fetch_serp.fetch_images("Arda Guler", target=20)

.env needs:  SERP_API_KEY=
"""
import os, sys, json, hashlib, pathlib, requests, itertools, time
from urllib.parse import urlencode, urlparse
from PIL import Image
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────────────────────
MIN_WIDTH  = 1000
BLOCKLIST  = {"lookaside.instagram.com", "lookaside.fbsbx.com", "img.uefa.com"}
HEADERS    = {"User-Agent": "Mozilla/5.0"}
OUT_DIR    = pathlib.Path("assets"); OUT_DIR.mkdir(exist_ok=True)

load_dotenv()
API_KEY = os.getenv("SERP_API_KEY")
if not API_KEY:
    sys.exit("❌  SERP_API_KEY missing in .env")

# ────────────────────────────────────────────────────────────────────────────
def _good_host(url: str) -> bool:
    return urlparse(url).hostname not in BLOCKLIST


def _save_image(url: str, meta: dict, tries: int = 3) -> str | None:
    """Download & verify; return local path or None."""
    fn = None  # ensure defined for cleanup
    for attempt in range(tries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            raw = resp.content

            fn = OUT_DIR / (hashlib.md5(raw).hexdigest() + ".jpg")
            fn.write_bytes(raw)

            # corruption + width check
            with Image.open(fn) as im:
                im.verify()
            with Image.open(fn) as im:
                if im.width < MIN_WIDTH:
                    raise ValueError(f"{im.width}px < {MIN_WIDTH}")

            json.dump(meta, open(fn.with_suffix(".json"), "w", encoding="utf-8"),
                      indent=2, ensure_ascii=False)
            print("✓", meta.get("title", url)[:60])
            return str(fn)

        except Exception as e:
            if fn and fn.exists():
                fn.unlink(missing_ok=True)
            if attempt == tries - 1:
                print("✗", url[:60], "→", e)
            else:
                time.sleep(1)

    return None


def _serpapi_hits(query: str):
    """Yield (url, meta) tuples from successive SerpAPI pages."""
    for page in itertools.count():
        params = {
            "engine":  "google_images",
            "q":       query,
            "ijn":     page,
            "num":     100,
            "tbs":     "isz:lt,islt:svga",
            "api_key": API_KEY,
        }
        data = requests.get("https://serpapi.com/search.json",
                            params=params, headers=HEADERS, timeout=20).json()
        for h in data.get("images_results", []):
            yield h["original"], {
                "title": h.get("title"),
                "width": h.get("width", 0),
                "attribution": h.get("link"),
                "src": h.get("link"),
            }
        if not data.get("images_results"):   # no more pages
            break

# ────────────────────────────────────────────────────────────────────────────
def fetch_images(query: str, target: int = 15) -> list[str]:
    """Download up to *target* valid images, return list of local paths."""
    print(f"\n🔍  Need {target} ≥{MIN_WIDTH}px images for: {query!r}\n")
    saved = []

    for url, meta in _serpapi_hits(query):
        if len(saved) >= target:
            break
        if _good_host(url):
            path = _save_image(url, meta)
            if path:
                saved.append(path)

    print(f"\n🎉  {len(saved)} image(s) saved in {OUT_DIR.resolve()}\n")
    return saved

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_serp.py \"search phrase\" [count]")
        sys.exit()
    phrase = sys.argv[1]
    tgt    = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    fetch_images(phrase, tgt)
