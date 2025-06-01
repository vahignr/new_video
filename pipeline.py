#!/usr/bin/env python
"""
pipeline.py – query → script → voiced segments → per-scene images → video
Updated to use batch image query processing and a refined topic for better context.
"""

import os
import re
import logging
from typing import List, Optional

from llm_processor import generate_script
from parser import parse_script
from tts_processor import TTSProcessor
from fetch_serp import fetch_images
from video_generator import VideoGenerator
from image_query_refiner import batch_refine_scenes

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s  [%(module)s:%(lineno)d] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
def _slug(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_-]+", "_", s).strip(" _")[:60]

def _fetch_one_image(query: str) -> Optional[str]:
    paths = fetch_images(query, 1)
    return paths[0] if paths else None

# ──────────────────────────────────────────────────────────────────────────
def build_video(
    query: str,
    refined_topic: str,  # <<< NEW ARGUMENT for the refined main subject
    images_per_segment: int = 2,
    voice: str = "nova",
    narration_style: str = "Friendly, upbeat narration.",
    output_filename: str = "final.mp4",
) -> str | None:

    # 1 ─ script & save
    log.info(f"Generating script for query: '{query}' with web research...")
    script_text, sources = generate_script(query) # Original query for script generation
    os.makedirs("output", exist_ok=True)
    txt_path = f"output/{_slug(query)}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(script_text)
    log.info(f"Script saved → {txt_path}")

    # 2 ─ segments
    log.info("Parsing script into segments...")
    segments = parse_script(script_text)
    if not segments:
        log.error("No segments parsed from script."); return None
    log.info(f"Parsed script into {len(segments)} segment(s)")

    # 3 ─ TTS
    log.info(f"Converting text to speech using voice '{voice}'...")
    tts = TTSProcessor(voice=voice, instructions=narration_style)
    voiced: List[dict] = []
    for idx, seg in enumerate(segments):
        mp3 = tts.scene_to_mp3(seg, idx)
        if mp3:
            seg["audio_path"] = mp3
            # 'duration' should ideally be set by TTSProcessor if it can determine it
            # If not, VideoGenerator might need to calculate it or have a default.
            # seg["duration"] = get_audio_duration(mp3) # Example
            voiced.append(seg)
        else:
            log.warning(f"TTS failed for segment {idx}. Skipping.")


    if not voiced:
        log.error("All TTS calls failed. Cannot proceed."); return None
    log.info(f"Successfully generated audio for {len(voiced)} segments.")

    # 4 ─ BATCH IMAGE QUERY GENERATION
    # Using the refined_topic for better contextual image queries
    log.info(f"Generating image queries for all segments using refined topic: '{refined_topic}'...")
    all_queries_dict = batch_refine_scenes(voiced, images_per_segment, topic=refined_topic)
    log.info("Image query generation complete.")
    
    # 5 ─ Fetch images based on batch-generated queries
    log.info("Fetching images for all segments...")
    for idx, seg_data in enumerate(voiced): # Iterate through the 'voiced' list which contains segment dicts
        # Get queries for the current segment index (which should be an int key)
        queries_for_segment = all_queries_dict.get(idx, [refined_topic] * images_per_segment) # Fallback to refined_topic
        
        seg_imgs: List[str] = []
        log.info(f"Segment {idx}: Attempting to fetch images for queries: {queries_for_segment}")
        for q_idx, q_str in enumerate(queries_for_segment):
            img_path = _fetch_one_image(q_str)
            if not img_path:
                log.warning(f"Segment {idx}, Query {q_idx} ('{q_str}'): Failed to fetch. Trying refined topic fallback.")
                # Fallback: try with refined topic + some variation
                img_path = _fetch_one_image(f"{refined_topic} visual {q_idx+1}")
            if img_path:
                seg_imgs.append(img_path)
            else:
                log.error(f"Segment {idx}, Query {q_idx} ('{q_str}'): All fetch attempts failed, including refined topic fallback.")
        
        # Ensure we have enough images per segment, using refined_topic for broad fallbacks
        while len(seg_imgs) < images_per_segment:
            log.warning(f"Segment {idx}: Not enough images ({len(seg_imgs)}/{images_per_segment}). Fetching more general fallback with refined topic.")
            fallback_img = _fetch_one_image(f"{refined_topic} photo {len(seg_imgs) + 1}") # Use refined_topic
            if fallback_img:
                seg_imgs.append(fallback_img)
            else:
                log.error(f"Segment {idx}: Could not fetch additional fallback images using refined topic '{refined_topic}'.")
                break # Avoid infinite loop if even broad fallbacks fail
        
        seg_data["images"] = seg_imgs # Add image paths to the segment dictionary
        log.info(f"Segment {idx}: {len(seg_imgs)} images finalized.")

    # 6 ─ video assembly
    log.info("Assembling final video...")
    vg  = VideoGenerator(output_dir="output", width=1920, height=1080) # Ensure your VideoGenerator can handle seg["duration"]
    mp4 = vg.create_video(voiced, output_filename=output_filename)
    return mp4

# ────────── tiny manual test ──────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate the refined topic for testing this module directly
    # In the full flow, main.py would call an LLM to get this.
    test_query = "advantages and disadvantages of hybrid cars"
    simulated_refined_topic = "Hybrid Cars" # This would come from the LLM in main.py
    
    log.info(f"Running test for pipeline.py with query: '{test_query}' and simulated refined topic: '{simulated_refined_topic}'")

    path = build_video(
        query=test_query,
        refined_topic=simulated_refined_topic, # Pass the simulated refined topic
        images_per_segment=1, # Reduced for faster testing
        voice="onyx", # Changed voice for variety in test
        narration_style="Clear and concise explanation.",
        output_filename="test_pipeline_output.mp4",
    )
    
    if path:
        log.info(f"\n✅ Pipeline test completed. Video saved to: {path}")
    else:
        log.error("\n✗ Pipeline test failed.")