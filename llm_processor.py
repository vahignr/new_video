#!/usr/bin/env python
"""
llm_processor.py  –  Enhanced script generator that handles diverse query types
and creates engaging YouTube-style content with proper web research integration.

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

ENHANCED_PROMPT = textwrap.dedent("""
    You are an expert YouTube content creator who produces engaging, well-researched videos.

    **RESEARCH FIRST**: Before writing, conduct thorough web searches to gather current, 
    accurate information. Use multiple searches to cover different aspects of the topic.

    **YOUR TASK**: Create a compelling ~1,200 word script for a YouTube video that will 
    keep viewers engaged throughout.

    **CONTENT ADAPTATION**: Adapt your style based on the query type:
    - **Sports/Football**: Energetic, match-focused, player highlights, league updates
    - **Music/Bands**: Passionate, concert atmosphere, album releases, band history
    - **News/Current Events**: Informative, factual, timeline-based, balanced coverage
    - **Top Lists**: Structured countdown, engaging descriptions, historical context
    - **General Topics**: Conversational, educational, broad appeal

    **HEADING STRUCTURE (CRITICAL)**:
    - Start with "### Introduction" (warm welcome, topic overview)
    - Use 3-6 INFORMATIVE section headings like:
      ✅ "### Real Madrid's Spectacular Season"
      ✅ "### Serie A: Juventus vs Inter Milan Rivalry" 
      ✅ "### Black Sabbath: Masters of Heavy Metal"
      ✅ "### Breaking: March 2025 Market Developments"
    - Avoid generic headings like "Current Situation", "Recent Developments"
    - End with "### Conclusion" (summary, call-to-action)
    - Finish with "### Sources:" (list all URLs you accessed)

    **WRITING STYLE**:
    - Conversational, engaging tone (like speaking to a friend)
    - Short paragraphs (2-4 sentences)
    - Use transition phrases between sections
    - Include specific facts, dates, names, numbers
    - No bullet points or markdown links in main content
    - Build excitement and maintain viewer interest

    **TECHNICAL REQUIREMENTS**:
    - ~1,200 words total
    - 6-8 main sections (including intro/conclusion)
    - Each section should be 100-200 words
    - End with complete source URLs (one per line, no formatting)

    **QUERY TO PROCESS**: "{query}"

    Remember: Research thoroughly, write engagingly, structure clearly, cite sources properly.
""").strip()

URL_REGEX = re.compile(r'https?://\S+')

# ────────── main helper ───────────────────────────────────────────────────
def generate_script(query: str,
                    target_words: int = TARGET_WORDS
                    ) -> Tuple[str, List[str]]:
    """
    Returns (script_text, list_of_source_urls).
    Enhanced to handle diverse query types with better research integration.
    """

    response = client.responses.create(
        model=MODEL,
        tools=[{"type": "web_search_preview"}],
        tool_choice={"type": "web_search_preview"},  # force at least one search
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
    # Test with different query types
    test_queries = [
        "Top 5 Premier League Transfers January 2025",
        "Black Sabbath greatest hits and legacy", 
        "What happened in tech world in last month"
    ]
    
    print("🧪 Testing enhanced script generator...\n")
    
    for i, demo_query in enumerate(test_queries[:1], 1):  # Test just first one
        print(f"Test {i}: {demo_query}")
        script, sources = generate_script(demo_query)
        
        # Show preview
        preview_length = 500
        print(f"\n--- Script Preview ({len(script)} chars) ---")
        print(script[:preview_length] + ("..." if len(script) > preview_length else ""))
        
        # Show sections count
        sections = re.findall(r'#{3}\s*(.+)', script)
        print(f"\n📋 Sections found ({len(sections)}):")
        for j, section in enumerate(sections[:5], 1):
            print(f"  {j}. {section}")
        if len(sections) > 5:
            print(f"  ... and {len(sections) - 5} more")
            
        # Show sources
        print(f"\n🔗 Sources captured: {len(sources)}")
        for url in sources[:3]:
            print(f"  • {url}")
        if len(sources) > 3:
            print(f"  ... and {len(sources) - 3} more")
            
        print(f"\n✅ Test {i} completed successfully!\n")
        break  # Only test first query for demo
        
    print("🎉 Enhanced LLM processor ready for diverse queries!")