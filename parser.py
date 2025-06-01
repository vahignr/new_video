#!/usr/bin/env python
"""
parser.py – split a GPT script into text segments with natural flow.
Each segment dict:
    {
        "type": "text",
        "heading": "Tesla's Battery Revolution",
        "content": "Paragraphs of content..."
    }

Updated to handle natural scripts without rigid Introduction/Conclusion structure.
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
        # First content before any heading - treat as opening segment
        segs.append({
            "type": "text", 
            "heading": "Opening", 
            "content": first
        })

    it = iter(parts[1:])  # skip intro part
    for head, body in zip(it, it):
        head_clean = head.strip()
        body_clean = body.strip()
        
        # Skip Sources section entirely
        if head_clean.lower().startswith("sources"):
            log.info("Skipping Sources section from processing")
            continue
            
        if not body_clean:
            continue
        
        # For natural scripts, all headings are meaningful content descriptors
        # so we include them in the speech for context
        content = f"{head_clean}. {body_clean}"
        
        segs.append({
            "type": "text", 
            "heading": head_clean, 
            "content": content
        })

    log.info("Parsed script into %d segment(s)", len(segs))
    return segs