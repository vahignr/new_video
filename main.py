#!/usr/bin/env python
"""
main.py – one-command driver for the video pipeline.
Just tweak the constants below and run:  python main.py
"""

from pipeline import build_video

# ─────────── configurable knobs ───────────────────────────────────────────
QUERY                = "Biggest football transfer rumours and how they would work out. Include kevin debruyne to napoli, haaland to barcelona,  wirtz to bayern, gyokeres to arsenal, leao to barcelona, reiknders to man city, rashford to barcelona, cristiano ronaldo jr. to real madrid"
IMAGES_PER_SEGMENT   = 2                      # set 1…5 as you like
VOICE_NAME           = "ash"                  # voices: nova, fable, alloy, …
NARRATION_STYLE      = "Speak in English, speak as economist."
OUTPUT_FILENAME      = "newsss.mp4"
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
