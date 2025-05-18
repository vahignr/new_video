#!/usr/bin/env python
"""
parser.py – split a GPT script into text segments.
Each segment dict:
    {
        "type": "text",
        "heading": "Black Sabbath",
        "content": "Black Sabbath. Paragraphs …"
    }
"""

import re, logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

HEAD_RE = re.compile(r"^\s*#{3}\s*(.+)$", re.M)

def parse_script(script: str):
    segs = []
    parts = HEAD_RE.split(script)

    # parts example: ["intro body", "Heading1", "body1", "Heading2", "body2", …]
    first = parts[0].strip()
    if first:
        segs.append({"type": "text", "heading": None, "content": first})

    it = iter(parts[1:])  # skip intro part
    for head, body in zip(it, it):
        body_clean = body.strip()
        if body_clean:
            full = f"{head.strip()}. {body_clean}"
            segs.append({"type": "text", "heading": head.strip(), "content": full})

    log.info("Parsed script into %d segment(s)", len(segs))
    return segs

# quick test
if __name__ == "__main__":
    demo = "### A\nfoo.\n\n### B\nbar."
    for s in parse_script(demo):
        print(s)
