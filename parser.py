#!/usr/bin/env python
"""
parser.py – split a GPT script into text segments.
Each segment dict:
    {
        "type": "text",
        "heading": "Black Sabbath",
        "content": "Black Sabbath. Paragraphs …"
    }

Updated to handle structural vs informative headings properly:
- Structural headings (Introduction, Conclusion, Sources) are not prepended to speech
- Informative headings (Serie A: Latest News) are prepended to speech
- Sources section is completely excluded from TTS processing
"""

import re, logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

HEAD_RE = re.compile(r"^\s*#{3}\s*(.+)$", re.M)

# Structural headings that shouldn't be spoken
STRUCTURAL_HEADINGS = {
    "introduction", "conclusion", "sources", "references", 
    "bibliography", "further reading", "credits"
}

def parse_script(script: str):
    segs = []
    parts = HEAD_RE.split(script)

    # parts example: ["intro body", "Heading1", "body1", "Heading2", "body2", …]
    first = parts[0].strip()
    if first:
        segs.append({"type": "text", "heading": None, "content": first})

    it = iter(parts[1:])  # skip intro part
    for head, body in zip(it, it):
        head_clean = head.strip()
        body_clean = body.strip()
        
        # Skip Sources section entirely
        if head_clean.lower().startswith("sources"):
            log.info("Skipping Sources section from TTS processing")
            continue
            
        if not body_clean:
            continue
            
        # Check if this is a structural heading that shouldn't be spoken
        is_structural = any(
            head_clean.lower().startswith(structural) 
            for structural in STRUCTURAL_HEADINGS
        )
        
        if is_structural:
            # Don't prepend structural headings to content
            content = body_clean
            log.info(f"Structural heading detected: '{head_clean}' - not prepending to speech")
        else:
            # Prepend informative headings to content
            content = f"{head_clean}. {body_clean}"
            log.info(f"Informative heading detected: '{head_clean}' - prepending to speech")
        
        segs.append({
            "type": "text", 
            "heading": head_clean, 
            "content": content
        })

    log.info("Parsed script into %d segment(s)", len(segs))
    return segs

# quick test
if __name__ == "__main__":
    demo = """### Introduction
Welcome to our show!

### Serie A: Latest News
Italian football is exciting.

### Conclusion
That's all for today.

### Sources
- https://example.com"""
    
    for s in parse_script(demo):
        print(f"Heading: {s['heading']}")
        print(f"Content: {s['content']}")
        print("---")