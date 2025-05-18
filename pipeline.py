#!/usr/bin/env python
"""
pipeline.py – query → script → voiced segments → per-scene images → video
"""

import os, re, logging
from typing import List, Optional

from llm_processor        import generate_script
from parser               import parse_script
from tts_processor        import TTSProcessor
from fetch_serp           import fetch_images
from video_generator      import VideoGenerator
from image_query_refiner  import refine_scene          # NEW

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s  %(message)s")
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
    images_per_segment: int = 2,
    voice: str = "nova",
    narration_style: str = "Friendly, upbeat narration.",
    output_filename: str = "final.mp4",
) -> str | None:

    # 1 ─ script & save
    script_text, sources = generate_script(query)
    os.makedirs("output", exist_ok=True)
    txt_path = f"output/{_slug(query)}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(script_text)
    print(f"\n📄 Script saved → {txt_path}")

    # 2 ─ segments
    segments = parse_script(script_text)
    if not segments:
        log.error("No segments parsed."); return None

    # 3 ─ TTS
    tts = TTSProcessor(voice=voice, instructions=narration_style)
    voiced: List[dict] = []
    for idx, seg in enumerate(segments):
        mp3 = tts.scene_to_mp3(seg, idx)
        if mp3:
            seg["audio_path"] = mp3
            voiced.append(seg)

    if not voiced:
        log.error("All TTS calls failed."); return None

    # 4 ─ per-scene image queries via LLM
    for seg in voiced:
        queries = refine_scene(seg["content"], images_per_segment, topic=query)
        seg_imgs = [ _fetch_one_image(q) or _fetch_one_image(query) for q in queries ]
        seg["images"] = seg_imgs

    # 5 ─ video assembly
    vg  = VideoGenerator(output_dir="output", width=1920, height=1080)
    mp4 = vg.create_video(voiced, output_filename=output_filename)
    return mp4

# ────────── tiny manual test ──────────────────────────────────────────────
if __name__ == "__main__":
    path = build_video(
        query="Top 10 Metal Bands of All Time",
        images_per_segment=2,
        voice="ash",
        narration_style="Energetic rock-style narration.",
        output_filename="top10_metal_bands.mp4",
    )
    print("\n✅ Done" if path else "\n✗ Pipeline failed")
