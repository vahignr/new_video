#!/usr/bin/env python
"""
tts_processor.py – speak text with gpt-4o-mini-tts, stripping URLs & citations.
"""

import os, re, time, logging
from typing import Optional, List
from dotenv import load_dotenv
from pydub import AudioSegment
from openai import OpenAI

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY missing")

MODEL = "gpt-4o-mini-tts"

class TTSProcessor:
    def __init__(self, temp_audio_dir="temp/audio",
                 voice="nova", instructions=""):
        self.client = OpenAI(api_key=API_KEY)
        self.dir = temp_audio_dir
        self.voice = voice
        self.inst = instructions or ""
        os.makedirs(self.dir, exist_ok=True)

    # ─────────── public ────────────────────────────────────────────────
    def text_to_mp3(self, text: str, filename: Optional[str] = None) -> Optional[str]:
        text = self._clean(text)
        if not filename:
            filename = f"tts_{int(time.time())}.mp3"
        if not filename.endswith(".mp3"):
            filename += ".mp3"
        out = os.path.join(self.dir, filename)

        if len(text) > 4096:
            return self._chunk_and_combine(text, out)

        try:
            with self.client.audio.speech.with_streaming_response.create(
                model=MODEL, voice=self.voice,
                input=text, instructions=self.inst,
                response_format="mp3",
            ) as resp:
                resp.stream_to_file(out)
            return out
        except Exception as e:
            log.error("TTS failed: %s", e)
            return None

    def scene_to_mp3(self, scene: dict, idx: int):
        return self.text_to_mp3(scene["content"], f"scene_{idx}.mp3")

    def duration_sec(self, mp3: str) -> float:
        try: return len(AudioSegment.from_mp3(mp3)) / 1000.0
        except: return 0.0

    # ─────────── cleaning ──────────────────────────────────────────────
    @staticmethod
    def _clean(txt: str) -> str:
        txt = txt.replace("###", "").replace("**", "").replace("#", "")
        txt = re.sub(r'https?://\S+', '', txt)                     # raw URLs
        txt = re.sub(r'\[[^\]]+]\([^)]*\)', '', txt)               # [label](url)
        txt = re.sub(r'\([^)]*\.(com|org|net|gov)[^)]*\)', '', txt, flags=re.I)
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        return ". ".join(lines)

    # ─────────── chunk/combine (unchanged) ─────────────────────────────
    def _chunk_and_combine(self, text: str, out_path: str) -> Optional[str]:
        sentences = text.replace("\n", " ").split(". ")
        chunks, cur = [], ""
        for s in sentences:
            if len(cur) + len(s) + 2 < 4000:
                cur += s + ". "
            else:
                chunks.append(cur); cur = s + ". "
        if cur: chunks.append(cur)

        parts: List[str] = []
        for i, chunk in enumerate(chunks):
            fn = os.path.join(self.dir, f"chunk_{i}_{int(time.time())}.mp3")
            try:
                with self.client.audio.speech.with_streaming_response.create(
                    model=MODEL, voice=self.voice, input=chunk,
                    instructions=self.inst, response_format="mp3"
                ) as r: r.stream_to_file(fn)
                parts.append(fn); time.sleep(0.5)
            except Exception as e: log.error("Chunk %d failed: %s", i, e)

        return self._combine(parts, out_path)

    def _combine(self, paths: List[str], out: str) -> Optional[str]:
        if not paths: return None
        combined = AudioSegment.empty()
        for p in paths:
            try: combined += AudioSegment.from_mp3(p)
            except: pass
        combined.export(out, format="mp3")
        for p in paths: os.remove(p)
        return out

# quick test
if __name__ == "__main__":
    tts = TTSProcessor()
    mp3 = tts.text_to_mp3("Black Sabbath. Visit (https://example.com) [link](https://foo.com).")
    print("MP3:", mp3)
