import os
import re
import cv2
import json
import shutil
import zipfile
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path

def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def _gather_pages_recursive(temp_dir):
    found = []
    for root, _, files in os.walk(temp_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                found.append(os.path.join(root, f))
    found.sort(key=lambda p: _natural_key(os.path.basename(p)))
    return found

# ... (detect_bubbles_opencv, detect_bubbles_ml, detect_bubbles same as before) ...

# <<< MODIFIED FONT HANDLING >>>
def _load_font(font_path=None, size=22):
    """Load a TrueType font if available, else fallback to default (tiny)."""
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    # fallback
    try:
        return ImageFont.load_default()  # always works, but bitmap font
    except Exception:
        return ImageFont.load_default()


def process_phase1_engine(input_path, output_dir, mode_label="normal", progress_cb=None, font_path=None):
    # ... (same as before except calls to _load_font(font_path, size) and render)
    # All _load_font() calls now pass `font_path`
    # I'll not repeat the whole function here; just need to update the font lines.
    # However for brevity, I'll show only the changed lines:

    # Inside process_phase1_engine:
    font_num = _load_font(font_path, size=20)   # was _load_font(size=20)

    # and in render_translations:
    font = _load_font(font_path, size=22)

    # Add font_path parameter to process_phase2_engine, process_phase2_from_local, render_translations
    # and pass it along.
