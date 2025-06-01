#!/usr/bin/env python
"""
image_query_refiner.py   ◂◂ o3-mini edition ▸▸
──────────────────────────────────────────────
Generate clean Google-Images search queries from text segments.

Public API
----------
• refine_headings(headings, topic)            -> {heading: query}
• refine_scene(text, n, topic)                -> [str]         (len == n)
• batch_refine_scenes(segments, k, topic)     -> {idx: [str]}  (len == k)

Internal helpers are all self-contained; drop this file in place of the old
version and re-run your pipeline.
"""

from __future__ import annotations

import os, re, json, logging, textwrap
from typing import List, Dict, Any

from openai import OpenAI

# ── basic logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s  %(levelname)s [image_query_refiner]  %(message)s",
)
log = logging.getLogger(__name__)

# ── OpenAI client ──────────────────────────────────────────────────────────────
_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=_api_key) if _api_key else None
O3_MODEL = "o3-mini"

# Pull { … } or [ … ] from a noisy reply
JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.S)

# ── helpers ────────────────────────────────────────────────────────────────────
def _get_topic_keyword(topic: str) -> str:
    """Return one concrete keyword to use in fallbacks."""
    if not topic or not topic.strip():
        return "image"
    stop = {
        "top", "the", "a", "an", "how", "to", "for", "of", "in",
        "will", "is", "are", "and", "advantages", "disadvantages",
    }
    words = [w.lower() for w in topic.split() if w.lower() not in stop]
    return sorted(words, key=len, reverse=True)[0] if words else topic.split()[0]


def _call_o3(prompt: str, *, expect_json: bool) -> str | None:
    """Fire one /responses request; return the raw string answer or None."""
    if not client:
        log.error("OpenAI client not initialised – set OPENAI_API_KEY.")
        return None
    try:
        fmt_type = "json_object" if expect_json else "text"
        resp = client.responses.create(
            model=O3_MODEL,
            input=[{"role": "user", "content": prompt}],
            text={"format": {"type": fmt_type}},
            reasoning={"effort": "medium", "summary": "auto"},
            store=False,
        )

        # New SDK (>= 2025-05) provides .output_text (str)
        if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
            return resp.output_text

        # Older/beta fallbacks
        if hasattr(resp, "text"):
            if isinstance(resp.text, str):              # direct str
                return resp.text
            if hasattr(resp.text, "value"):             # ResponseTextConfig
                return resp.text.value
        return None
    except Exception as e:
        log.error(f"O3 call failed: {e}")
        return None


# ── master prompt logic (unchanged) ────────────────────────────────────────────
MASTER_PROMPT_CORE_LOGIC = textwrap.dedent("""
You are an AI expert at generating simple, visual Google Images search queries that return clean, high-quality images without text overlays, charts, or infographics.

**MAIN TOPIC**: "{topic}"

## CORE PRINCIPLE: ALWAYS SHOW THE MAIN OBJECT
**CRITICAL RULE**: When the main topic is a tangible object (like "Hybrid Cars"), ALWAYS query for that exact object, regardless of what aspect the segment discusses.

## QUERY GENERATION STRATEGY:

### 1. FOR TANGIBLE OBJECTS (Cars, Phones, Food, etc.)
**ALWAYS use the main object name**, even if segment discusses concepts about it:
- Main Topic: "Hybrid Cars" 
  - Segment about performance → "Hybrid Cars"
  - Segment about benefits → "Hybrid Cars" 
  - Segment about costs → "Hybrid Cars"
  - Segment about choosing → "Hybrid Cars"
- Main Topic: "iPhone"
  - Segment about features → "iPhone"
  - Segment about price → "iPhone"

### 2. ADD SPECIFIC MODELS/BRANDS (SECONDARY OPTION)
If you need variety, use specific popular models/brands:
- "Toyota Prius", "Honda Accord Hybrid", "Tesla Model 3"
- "iPhone 15", "Samsung Galaxy", "MacBook Pro"

### 3. FOR PEOPLE/TEAMS
Use exact names: "Messi", "Real Madrid", "Kevin De Bruyne"

### 4. FOR ABSTRACT TOPICS
Find the most concrete, photographable representation:
- Economic Trends → "Stock Market", "Money", "Wall Street"
- Dreams → "Sleeping", "Bedroom"
- Music → "Guitar", "Concert", "Recording Studio"

### 5. ABSOLUTELY AVOID
**NEVER use these concept words that generate charts/infographics:**
- Performance, Benefits, Advantages, Disadvantages
- Efficiency, Cost, Price, Choosing, Comparison
- Analysis, Statistics, Trends, Data
- Tips, Guide, How-to, Steps

### 6. TEMPORAL CONTEXT
Only add years for time-sensitive topics: "Hybrid Cars 2025"

## EXAMPLES BY TOPIC TYPE:

**Sports**: "Messi", "Real Madrid", "Champions League Trophy"
**Technology**: "iPhone", "Quantum Computer", "AI Robot"
**Food**: "Sesame Chicken", "Pizza", "Sushi"
**Economics**: "Stock Market", "Money", "Bitcoin"
**Music**: "Guitar", "Concert Stage", "Recording Studio"
**Health**: "Fitness", "Yoga", "Running"
**Places**: "Chicago", "Tokyo", "Mountains"

## OUTPUT FORMAT:
Provide only the search query, nothing else.
""").strip()

# ── prompt templates ───────────────────────────────────────────────────────────
BATCH_SCENE_PROMPT_TEMPLATE = textwrap.dedent("""
{master_core_logic}

YOUR TASK FOR BATCH PROCESSING:
The MAIN TOPIC is already defined above. Analyse each segment below.
For each segment, generate exactly {images_per_segment} distinct image search queries.

SEGMENTS TO PROCESS (JSON):
{segments_json}

Return one JSON object:
  {{ "0": ["q1","q2"], "1": ["q3","q4"] }}
""").strip()

SCENE_PROMPT_TEMPLATE = textwrap.dedent("""
{master_core_logic}

YOUR TASK FOR SINGLE SCENE PROCESSING:
Generate exactly {n} image search queries for the paragraph below.

PARAGRAPH:
"{text}"

Return a JSON array (len == {n}).
""").strip()

HEAD_PROMPT_TEMPLATE = textwrap.dedent("""
{master_core_logic}

YOUR TASK FOR HEADING REFINEMENT:
For each heading below, give ONE image query.

HEADINGS (JSON list):
{headings_json}

Return a JSON object mapping heading → query.
""").strip()

# ── public functions ───────────────────────────────────────────────────────────
def batch_refine_scenes(
    segments: List[Dict[str, Any]], images_per_segment: int, topic: str
) -> Dict[int, List[str]]:
    topic_kw = _get_topic_keyword(topic)

    seg_json = json.dumps(
        [
            {
                "index": idx,
                "heading": str(s.get("heading", "")),
                "content": str(s.get("content", ""))[:300],
            }
            for idx, s in enumerate(segments)
        ],
        indent=2,
    )

    prompt = BATCH_SCENE_PROMPT_TEMPLATE.format(
        master_core_logic=MASTER_PROMPT_CORE_LOGIC.format(topic=topic),
        images_per_segment=images_per_segment,
        segments_json=seg_json,
    )

    raw = _call_o3(prompt, expect_json=True)
    if raw is None:
        return {
            i: [f"{topic_kw} segment {i} fallback {j+1}" for j in range(images_per_segment)]
            for i in range(len(segments))
        }

    try:
        data: Dict[str, List[str]] = json.loads(raw)
        out: Dict[int, List[str]] = {}
        for i in range(len(segments)):
            qlist = data.get(str(i), [])
            qlist = [q for q in qlist if isinstance(q, str)]
            qlist = (qlist + [f"{topic_kw} segment {i} extra"] * images_per_segment)[
                :images_per_segment
            ]
            out[i] = qlist
        return out
    except Exception as e:
        log.error(f"batch_refine_scenes JSON parse error: {e}")
        return {
            i: [f"{topic_kw} error {j+1}" for j in range(images_per_segment)]
            for i in range(len(segments))
        }


def refine_scene(text: str, n: int, topic: str) -> List[str]:
    topic_kw = _get_topic_keyword(topic)

    prompt = SCENE_PROMPT_TEMPLATE.format(
        master_core_logic=MASTER_PROMPT_CORE_LOGIC.format(topic=topic),
        text=text[:600],
        n=n,
    )

    raw = _call_o3(prompt, expect_json=True)
    if raw is None:
        return [f"{topic_kw} placeholder {i+1}" for i in range(n)]

    try:
        json_str = JSON_RE.search(raw).group(0) if JSON_RE.search(raw) else raw
        lst: List[str] = json.loads(json_str)
        if not (isinstance(lst, list) and all(isinstance(x, str) for x in lst)):
            raise ValueError("bad structure")
        return (lst + [f"{topic_kw} extra"] * n)[:n]
    except Exception as e:
        log.error(f"refine_scene parse error: {e}")
        return [f"{topic_kw} error {i+1}" for i in range(n)]


def refine_headings(headings: List[str], topic: str) -> Dict[str, str]:
    if not headings:
        return {}
    topic_kw = _get_topic_keyword(topic)

    prompt = HEAD_PROMPT_TEMPLATE.format(
        master_core_logic=MASTER_PROMPT_CORE_LOGIC.format(topic=topic),
        headings_json=json.dumps(headings),
    )

    raw = _call_o3(prompt, expect_json=True)
    if raw is None:
        return {h: f"{topic_kw} placeholder" for h in headings}

    try:
        data: Dict[str, str] = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not a dict")
        return {
            h: data[h] if isinstance(data.get(h), str) and data[h].strip() else f"{topic_kw} {h[:15]} fallback"
            for h in headings
        }
    except Exception as e:
        log.error(f"refine_headings parse error: {e}")
        return {h: f"{topic_kw} error" for h in headings}


# ── basic self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_segments = [
        {"heading": "Hybrid Car Performance", "content": "Hybrids deliver instant torque …"},
        {"heading": "Hybrid Car Costs", "content": "Up-front price v/s long-term savings …"},
    ]
    print(batch_refine_scenes(demo_segments, 2, "Hybrid Cars"))
    print(refine_scene("The iPhone 15 introduces a periscope lens …", 3, "iPhone"))
    print(refine_headings(["Messi", "Real Madrid", "Quantum Computing"], "Sports & Tech"))
