#!/usr/bin/env python
"""
image_query_refiner.py
──────────────────────
Enhanced LLM helpers to turn headings or full scene paragraphs into specific
Google-Images search queries that prioritize current, engaging visuals.

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

# Initialize client only if API key is available
_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=_api_key) if _api_key else None
JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.S)

# ── 1. Batch headings → single query each ──────────────────────────────────
HEAD_PROMPT = """
You convert slide headings into engaging Google Images queries that will find 
visually compelling content for video backgrounds.

IMPORTANT GUIDELINES:
• AVOID logos, crests, emblems, or static graphics
• PREFER action shots, match photos, crowd scenes, player celebrations
• For sports: use "match photo", "action shot", "celebration", "stadium crowd"
• For music: use "live concert", "band photo", "performance shot", "festival crowd"
• For news/topics: use "crowd photo", "scene photo", "event photo"

For specific entities (teams, players, bands): add visual context like:
- "Real Madrid match action photo" instead of "Real Madrid logo"
- "Messi celebration photo" instead of "Messi"
- "Black Sabbath live concert" instead of "Black Sabbath"

For generic headings (Introduction, Conclusion): combine the MAIN TOPIC "{topic}" 
with a crowd/action visual, e.g. "{topic} stadium crowd photo".

Return ONLY a JSON object mapping each heading to its query — no prose.
Headings:
{headings}
""".strip()

def refine_headings(headings: List[str], topic: str) -> Dict[str, str]:
    if not client:
        log.warning("OpenAI client not initialized - using fallback")
        return {h: f"{topic} action photo" for h in headings}
    
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
        return {h: f"{topic} action photo" for h in headings}

# ── 2. Scene paragraph → N queries in order ───────────────────────────────
SCENE_PROMPT = textwrap.dedent("""
You are generating Google Images search queries for video backgrounds by analyzing paragraph content.

STEP 1: ANALYZE the paragraph content and theme
- What is this paragraph specifically about?
- What are the key concepts being discussed?
- What visual elements would best represent this content?

STEP 2: GENERATE queries with these rules:
• Never use bare entity names - always include visual + temporal context
• Always add year (2024/2025) or "recent" for current images
• Add current team/context when mentioned
• Make queries specific to the paragraph's theme/content

EXAMPLES by paragraph theme:

📈 DOMINANCE/SUCCESS paragraph:
❌ "Real Madrid 2024"
✅ "Real Madrid victory celebration 2024" or "Real Madrid trophy lift 2024"

⚽ TACTICAL/STRATEGY paragraph:
❌ "Real Madrid match 2024"
✅ "Real Madrid tactical formation 2024" or "Real Madrid training tactics 2024"

👨‍💼 PLAYER paragraph:
❌ "Messi 2024"
✅ "Messi Inter Miami training 2024" or "Messi Inter Miami press conference 2024"

🏟️ TOURNAMENT/VENUE paragraph:
❌ "FIFA Club World Cup 2025"
✅ "FIFA Club World Cup 2025 stadium atmosphere" or "Club World Cup 2025 fan crowd"

🎯 CHALLENGES/PROBLEMS paragraph:
❌ "Inter Miami 2024"
✅ "Inter Miami tactical difficulties 2024" or "Inter Miami defensive pressure 2024"

YOUR TASK:
1. Read and understand what this specific paragraph discusses
2. Identify {n} key entities/concepts from the paragraph
3. For each, create a query that matches the paragraph's theme + current context
4. Keep queries 3-5 words maximum
5. Ensure each query is specific to the paragraph content, not generic

Return exactly {n} queries as JSON array.

Paragraph to analyze:
"{text}"
""").strip()

def refine_scene(text: str, n: int, topic: str) -> List[str]:
    if not client:
        log.warning("OpenAI client not initialized - using fallback")
        return [f"{topic} photo"] * n
    
    try:
        msg = SCENE_PROMPT.format(text=text[:1000], n=n)
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
    if not _api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping test.")
        print("Usage: Set OPENAI_API_KEY in .env file and run again.")
    else:
        # Test headings 
        heads = ["Introduction", "Real Madrid vs Barcelona", "Conclusion"]
        print("Headings →", refine_headings(heads, topic="La Liga Week Summary"))

        # Test scene with different paragraph types
        tactical_paragraph = """
        Real Madrid's tactical approach revolves around high pressing and quick transitions. 
        Their midfield, orchestrated by Luka Modrić and Casemiro, controls the tempo, 
        allowing for swift counter-attacks against opponents.
        """
        print("Tactical Scene →", refine_scene(tactical_paragraph, 2, topic="Real Madrid Tactics"))
        
        player_paragraph = """
        For Real Madrid, players like Kylian Mbappé, who netted 22 goals in La Liga 
        this season, are pivotal. On the other hand, Inter Miami's Lionel Messi, 
        despite his age, continues to showcase his playmaking abilities.
        """
        print("Player Scene →", refine_scene(player_paragraph, 2, topic="Key Players"))