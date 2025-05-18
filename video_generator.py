#!/usr/bin/env python
"""
video_generator.py – stitch voiced segments + images into an MP4.
No text overlays; only images + audio.
"""

import os, logging
from typing import List
import numpy as np
from PIL import Image
from tqdm import tqdm

# Pillow 10 compatibility for MoviePy resize
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from moviepy.editor import (
    AudioFileClip, ImageClip, ColorClip,
    CompositeVideoClip, concatenate_videoclips, vfx
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class VideoGenerator:
    def __init__(self, output_dir="output", width=1920, height=1080, fps=30):
        self.out_dir = output_dir
        self.W, self.H = width, height
        self.fps = fps
        os.makedirs(self.out_dir, exist_ok=True)

    # ───────────────────────────────────────────────────────────────
    def create_video(self, segments: List[dict], output_filename="out.mp4"):
        clips = []
        log.info("Building video segments …")
        for seg_idx, seg in enumerate(tqdm(segments, desc="Segments")):
            audio_path = seg.get("audio_path")
            if not audio_path or not os.path.exists(audio_path):
                continue
            audio = AudioFileClip(audio_path)
            dur   = audio.duration

            img_paths = [p for p in (seg.get("images") or []) if p and os.path.exists(p)]
            if not img_paths:
                img_clip = ColorClip((self.W, self.H), color=(20, 20, 20)).set_duration(dur)
                img_clip = img_clip.set_audio(audio)
                clips.append(img_clip)
                continue

            per = dur / len(img_paths)
            subclips = []
            for j, path in enumerate(img_paths):
                ic = self._img_clip(path, per)
                ic = ic.fx(vfx.colorx, 1.1).fx(vfx.resize, lambda t: 1 + 0.04 * t / per)
                ic = ic.set_audio(audio.subclip(j * per, (j + 1) * per))
                subclips.append(ic)

            # cross-fade between images of the same segment
            if len(subclips) > 1:
                subclips = [c.crossfadein(0.5) if i else c for i, c in enumerate(subclips)]
            seg_video = concatenate_videoclips(subclips, method="compose")
            clips.append(seg_video)

        if not clips:
            log.error("No clips created.")
            return None

        # cross-fade between segments
        final_clips = []
        for i, c in enumerate(clips):
            if i:  c = c.crossfadein(0.5)
            if i < len(clips) - 1:
                c = c.crossfadeout(0.5)
            final_clips.append(c)

        final = concatenate_videoclips(final_clips, method="compose").set_fps(self.fps)
        out   = os.path.join(self.out_dir, output_filename)
        log.info("Encoding MP4 …")

        final.write_videofile(
            out,
            codec="libx264",
            audio_codec="aac",
            fps=self.fps,
            bitrate="6000k",
            preset="ultrafast",
            threads=4,
            logger="bar"        # MoviePy progress bar
        )
        final.close()
        return out

    # ─────────────────────────────────────────── helpers
    def _img_clip(self, path, duration):
        img = Image.open(path).convert("RGB")
        w0, h0 = img.size
        aspect_s = w0 / h0
        aspect_t = self.W / self.H
        if aspect_s > aspect_t:
            nw = self.W; nh = int(self.W / aspect_s)
        else:
            nh = self.H; nw = int(self.H * aspect_s)
        bg = Image.new("RGB", (self.W, self.H), (0, 0, 0))
        bg.paste(img.resize((nw, nh), Image.ANTIALIAS),
                 ((self.W - nw) // 2, (self.H - nh) // 2))
        return ImageClip(np.array(bg)).set_duration(duration).set_fps(self.fps)
