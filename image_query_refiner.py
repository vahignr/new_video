#!/usr/bin/env python
"""
image_query_refiner.py
──────────────────────
LLM helpers to turn headings or full scene paragraphs into concise
Google-Images search queries.

Public functions
----------------
• refine_headings(headings, topic)  -> dict{heading: query}
• refine_scene(text, n, topic)      -> list[str]  (len == n)

Both rely on gpt-4o-mini and guarantee JSON-parsable output or fallback.
"""

from __future__ import annotations
import os, re, json, logging, textwrap
from typing import List, Dict
from openai import OpenAI

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.S)

# ── 1. Batch headings → single query each ──────────────────────────────────
HEAD_PROMPT = """
You convert slide headings into concise Google Images queries (3-6 words).

* Keep specific names (e.g. "Black Sabbath") + a visual hint ("live photo").
* For generic headings like Introduction, Conclusion, Sources, combine the
  MAIN TOPIC "{topic}" with a visual phrase, e.g. "{topic} crowd photo".

Return ONLY a JSON object mapping each heading to its query — no prose.
Headings:
{headings}
""".strip()

def refine_headings(headings: List[str], topic: str) -> Dict[str, str]:
    try:
        msg = HEAD_PROMPT.format(headings=json.dumps(headings), topic=topic)
        r   = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                messages=[{"role": "system", "content": msg}],
             )
        raw = r.choices[0].message.content
        m = JSON_RE.search(raw)
        data = json.loads(m.group(0)) if m else {}
        if not isinstance(data, dict): raise ValueError("bad JSON")
        return data
    except Exception as e:
        log.warning("Heading refiner fallback (%s)", e)
        return {h: f"{topic} photo" for h in headings}

# ── 2. Scene paragraph → N queries in order ───────────────────────────────
SCENE_PROMPT = textwrap.dedent("""
You are given a paragraph from a video script and an integer N.

• Identify the first N distinct, image-worthy entities *in order* of appearance
  (band, club, person, place, object, etc.).
• Return exactly N Google-Images queries (3-6 words each).  If fewer than N
  distinct entities exist, pad the list with queries that combine the MAIN
  TOPIC "{topic}" and a suitable visual phrase (e.g. "{topic} crowd photo").

Respond with a JSON array of strings, nothing else.

N = {n}

Paragraph:
\"\"\"{text}\"\"\"
""").strip()

def refine_scene(text: str, n: int, topic: str) -> List[str]:
    try:
        msg = SCENE_PROMPT.format(text=text[:1500], n=n, topic=topic)
        r   = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                messages=[{"role": "system", "content": msg}],
             )
        raw = r.choices[0].message.content
        m   = JSON_RE.search(raw)
        data = json.loads(m.group(0)) if m else []
        if not isinstance(data, list): raise ValueError("bad JSON array")
        data = [str(q).strip() for q in data][:n]
    except Exception as e:
        log.warning("Scene refiner fallback (%s)", e)
        data = []

    # pad / trim deterministically to length n
    if len(data) < n:
        pad = f"{topic} photo"
        data.extend([pad] * (n - len(data)))
    return data[:n]

# ── tiny manual test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    heads = ["Introduction", "Black Sabbath", "Conclusion"]
    print("Headings →", refine_headings(heads, topic="Top 10 Metal Bands"))

    paragraph = """
    Serie A has seen dramatic twists this week. AC Milan sacked Stefano Pioli;
    Napoli appointed Antonio Conte; Juventus lifted the Coppa Italia.
    """
    print("Scene →", refine_scene(paragraph, 2, topic="Serie A Weekly News"))
