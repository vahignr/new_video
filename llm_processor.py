#!/usr/bin/env python
"""
llm_processor.py  –  generate a ~1 200-word YouTube script that *must*
perform at least one web search and then list its sources.

Needs in .env:
    OPENAI_API_KEY=...

No SerpAPI key required.
"""

import os, sys, re, textwrap, logging
from typing import List, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# ────────── setup ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

load_dotenv()
client = OpenAI()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("❌  OPENAI_API_KEY missing in .env")

MODEL        = "gpt-4o-mini"
TARGET_WORDS = 1200

PROMPT = textwrap.dedent("""
    You are a seasoned YouTube script writer.

    **Before you write**, perform at least one web search to collect
    up-to-date facts.  Then craft an engaging article-style script of
    about 1 200 words (≈10 min when read aloud).

    • Conversational tone, short paragraphs.
    • Insert section headings starting with '###'.
    • No bullet lists or markdown links.
    • Finish with a section titled "Sources:" listing every URL you opened.

    QUERY: "{query}"
""").strip()

URL_REGEX = re.compile(r'https?://\S+')

# ────────── main helper ───────────────────────────────────────────────────
def generate_script(query: str,
                    target_words: int = TARGET_WORDS
                    ) -> Tuple[str, List[str]]:
    """
    Returns (script_text, list_of_source_urls).
    """

    response = client.responses.create(
        model=MODEL,
        tools=[{"type": "web_search_preview"}],
        tool_choice={"type": "web_search_preview"},  # force at least one search
        input=PROMPT.format(query=query),
    )

    script_text = response.output_text.strip()

    # Extract URLs from the "Sources:" block (or anywhere) via regex
    urls = URL_REGEX.findall(script_text)
    urls = list(dict.fromkeys(urls))  # dedupe, preserve order

    log.info("Script chars: %d   sources found: %d",
             len(script_text), len(urls))
    return script_text, urls

# ────────── tiny manual test ─────────────────────────────────────────────
if __name__ == "__main__":
    demo = "Last 1 Week Market News"
    script, sources = generate_script(demo)

    print("\n--- script preview ---\n")
    print(script[:400] + ("…" if len(script) > 400 else ""))
    print(f"\nSources captured: {len(sources)}")
    for u in sources[:6]:
        print("  •", u)
    print("\n✓ LLM processor (Responses API) OK\n")
