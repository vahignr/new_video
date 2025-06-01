#!/usr/bin/env python
"""
llm_processor.py  –  Enhanced script generator that creates natural, flowing content
without rigid structural sections and always uses web research for current information.

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

MODEL        = "gpt-4.1-2025-04-14"
TARGET_WORDS = 1200

ENHANCED_PROMPT = textwrap.dedent("""
    You are an expert YouTube content creator who produces engaging, well-researched videos.
    
    **CRITICAL: ALWAYS START WITH WEB SEARCHES**
    Before writing ANYTHING, you MUST conduct extensive web searches to gather:
    - Current news and developments (last 30 days if possible)
    - Latest statistics and data
    - Recent events and updates
    - Trending discussions about the topic
    
    **YOUR TASK**: Create a compelling ~1,200 word script that feels like natural storytelling,
    not a formal essay with rigid sections.

    **NATURAL FLOW GUIDELINES**:
    - NO formulaic "Introduction" or "Conclusion" sections
    - Start with an engaging hook, interesting fact, or current event
    - Let topics flow naturally from one to another
    - Use conversational transitions between ideas
    - End with forward-looking insights or thought-provoking questions
    
    **SECTION HEADINGS**:
    - Use 4-6 DESCRIPTIVE headings that preview content, NOT generic labels
    - Good: "### Tesla's Game-Changing Battery Breakthrough"
    - Good: "### Why Hybrid Cars Are Outselling EVs in 2024" 
    - Bad: "### Introduction", "### Overview", "### Conclusion"
    - Each heading should make viewers curious about what's next
    
    **CONTENT REQUIREMENTS**:
    - Open with something surprising, current, or attention-grabbing
    - Weave in specific dates, numbers, and recent examples throughout
    - Use short paragraphs (2-4 sentences) for easy listening
    - Include real-world examples and case studies
    - Build narrative tension - what's at stake? why does this matter now?
    
    **WRITING STYLE**:
    - Conversational and engaging, like talking to a curious friend
    - Use "you" to address the viewer directly
    - Include rhetorical questions to maintain engagement
    - Add personality - surprise, excitement, concern where appropriate
    - Avoid academic or overly formal language
    
    **TECHNICAL REQUIREMENTS**:
    - ~1,200 words total
    - 4-6 sections with descriptive headings
    - Each section: 200-300 words
    - Must cite sources at the end
    
    **IMPORTANT**: Research thoroughly first, then write naturally. The script should
    feel like an engaging story about "{query}", not a Wikipedia article.
    
    End with "### Sources:" followed by actual URLs you accessed (no formatting).
""").strip()

URL_REGEX = re.compile(r'https?://\S+')

# ────────── main helper ───────────────────────────────────────────────────
def generate_script(query: str,
                    target_words: int = TARGET_WORDS
                    ) -> Tuple[str, List[str]]:
    """
    Returns (script_text, list_of_source_urls).
    Enhanced to create natural, flowing content with mandatory web research.
    """

    response = client.responses.create(
        model=MODEL,
        tools=[{"type": "web_search_preview"}],
        tool_choice={"type": "web_search_preview"},  # force web search
        input=ENHANCED_PROMPT.format(query=query),
    )

    script_text = response.output_text.strip()

    # Extract URLs from the "Sources:" block (or anywhere) via regex
    urls = URL_REGEX.findall(script_text)
    urls = list(dict.fromkeys(urls))  # dedupe, preserve order

    log.info("Script generated - chars: %d, sections: ~%d, sources: %d",
             len(script_text), 
             len(re.findall(r'#{3}', script_text)),
             len(urls))
    
    return script_text, urls

# ────────── tiny manual test ─────────────────────────────────────────────
if __name__ == "__main__":
    demo_query = "advantages and disadvantages of hybrid cars"
    print(f"Generating script for: {demo_query}\n")
    
    script, sources = generate_script(demo_query)
    
    # Show preview
    print("--- Script Preview ---")
    print(script[:800] + "...\n")
    
    # Show sections
    sections = re.findall(r'#{3}\s*(.+)', script)
    print(f"Sections found ({len(sections)}):")
    for i, section in enumerate(sections, 1):
        print(f"  {i}. {section}")
    
    print(f"\nSources: {len(sources)}")
    print("✅ Script generated successfully!")