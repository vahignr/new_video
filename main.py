#!/usr/bin/env python
"""
main.py – one-command driver for the video pipeline.
Just tweak the constants below and run:  python main.py
Includes LLM-based topic refinement.
"""
import os
import logging
from openai import OpenAI
from pipeline import build_video # Assuming build_video is in pipeline.py

# Configure basic logging for this script
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [main] %(message)s")
log = logging.getLogger(__name__)

# Initialize OpenAI client (only if API key is available)
_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=_api_key) if _api_key else None

# ─────────── configurable knobs ───────────────────────────────────────────
QUERY                = "Advantages and Disadvantages of Hybrid Cars"
IMAGES_PER_SEGMENT   = 2
VOICE_NAME           = "ash"
NARRATION_STYLE      = "Speak in English, speak as sales person."
OUTPUT_FILENAME      = "hybrid_refined_topic.mp4"
# ───────────────────────────────────────────────────────────────────────────

TOPIC_REFINEMENT_PROMPT = """
You are an expert at identifying the core, tangible subject from a longer phrase or question.
Your goal is to extract a concise noun phrase (2-4 words) that represents the main visualizable subject.
Avoid abstract concepts like "advantages," "disadvantages," "reasons," "impact," etc., if a more concrete subject is present.

Examples:
- Phrase: "Advantages and Disadvantages of Hybrid Cars" -> Subject: "Hybrid Cars"
- Phrase: "Top 5 Tourist Destinations in Italy" -> Subject: "Italy Tourism" or "Italian Landmarks"
- Phrase: "The Impact of AI on Modern Art" -> Subject: "AI in Art" or "AI Generated Art"
- Phrase: "Exploring the Deepest Parts of the Ocean" -> Subject: "Deep Sea Exploration" or "Ocean Depths"
- Phrase: "Understanding Quantum Entanglement for Beginners" -> Subject: "Quantum Entanglement"
- Phrase: "Best Dog Breeds for Families with Kids" -> Subject: "Family Dog Breeds"
- Phrase: "Gyokeres to Arsenal ⚡️" -> Subject: "Gyokeres Arsenal"
- Phrase: "Lame La Liga – “Real Madrid Hating” Barcelona vs. “We won 15 UCLs” Real Madrid." -> Subject: "Real Madrid Barcelona" or "La Liga"

Phrase to analyze: "{raw_query}"

Output ONLY the concise core subject. Do not add any explanation or surrounding text.
Core Subject:
"""

def refine_query_to_main_topic(raw_query: str) -> str:
    """
    Uses an LLM to refine a raw query string into a concise main topic.
    """
    if not client:
        log.warning("OpenAI client not initialized for topic refinement. Using raw query as topic.")
        return raw_query # Fallback to raw query if no client

    if not raw_query or not raw_query.strip():
        log.warning("Raw query for topic refinement is empty. Returning 'General Topic'.")
        return "General Topic" # Fallback for empty query

    try:
        prompt_payload = TOPIC_REFINEMENT_PROMPT.format(raw_query=raw_query)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Or another suitable model
            temperature=0.1,    # Low temperature for more deterministic output
            messages=[
                {"role": "user", "content": prompt_payload}
            ],
            max_tokens=20 # Expecting a short response
        )
        
        refined_topic = response.choices[0].message.content.strip()
        
        # Basic cleaning: remove potential "Core Subject:" prefix if LLM includes it
        if refined_topic.lower().startswith("core subject:"):
            refined_topic = refined_topic[len("core subject:"):].strip()
        
        if not refined_topic: # If LLM returns empty string
            log.warning(f"LLM returned empty string for topic refinement from '{raw_query}'. Using raw query.")
            return raw_query
            
        log.info(f"Refined query '{raw_query}' to topic: '{refined_topic}'")
        return refined_topic

    except Exception as e:
        log.error(f"Error during topic refinement for '{raw_query}': {e}. Using raw query as topic.")
        return raw_query # Fallback to raw query on error


if __name__ == "__main__":
    if not _api_key:
        log.error("OPENAI_API_KEY not set. Topic refinement and subsequent LLM calls might fail or use fallbacks.")
        # Decide if you want to exit or proceed with potential fallbacks
        # exit(1) 

    # 1. Refine the initial QUERY to get a main topic
    log.info(f"Starting topic refinement for initial query: '{QUERY}'")
    refined_topic_for_video = refine_query_to_main_topic(QUERY)

    # 2. Call build_video with the refined topic
    log.info(f"Proceeding to build video with main topic: '{refined_topic_for_video}'")
    mp4_path = build_video(
        query=QUERY, # Original query might still be used for script generation, etc.
        refined_topic=refined_topic_for_video, # NEW: Pass the refined topic
        images_per_segment=IMAGES_PER_SEGMENT,
        voice=VOICE_NAME,
        narration_style=NARRATION_STYLE,
        output_filename=OUTPUT_FILENAME
    )

    if mp4_path:
        print(f"\n✅ Video ready → {mp4_path}")
    else:
        print("\n✗ Video pipeline failed")