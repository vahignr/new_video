#!/usr/bin/env python
"""
main.py – one-command driver for the video pipeline.
Just tweak the constants below and run:  python main.py
"""

from pipeline import build_video

# ─────────── configurable knobs ───────────────────────────────────────────
QUERY                = "Generate a youtube text script for latest new for football leagues like la-liga, italian, turkish, england etc. popular ones"
IMAGES_PER_SEGMENT   = 1                      # set 1…5 as you like
VOICE_NAME           = "ash"                  # voices: nova, fable, alloy, …
NARRATION_STYLE      = "Energetic football speaker narration."
OUTPUT_FILENAME      = "top10_metal_bands.mp4"
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp4_path = build_video(
        query=QUERY,
        images_per_segment=IMAGES_PER_SEGMENT,
        voice=VOICE_NAME,
        narration_style=NARRATION_STYLE,
        output_filename=OUTPUT_FILENAME
    )

    if mp4_path:
        print(f"\n✅ Video ready → {mp4_path}")
    else:
        print("\n✗ Video pipeline failed")
