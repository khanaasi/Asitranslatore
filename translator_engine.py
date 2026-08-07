import os
import re
import cv2
import json
import shutil
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path


def _natural_key(s):
    """Sorts '2.jpg' before '10.jpg'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def _gather_pages_recursive(temp_dir):
    """Recursively find every image inside temp_dir, sorted naturally."""
    found = []
    for root, _, files in os.walk(temp_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                found.append(os.path.join(root, f))
    found.sort(key=lambda p: _natural_key(os.path.basename(p)))
    return found


def detect_bubbles_opencv(image_path):
    """OpenCV Contour-based bubble detection (fallback)."""
    img = cv2.imread(image_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    h_img, w_img = gray.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue
        area = w * h
        aspect_ratio = float(w) / h
        if (0.002 * w_img * h_img) < area < (0.25 * w_img * h_img):
            if 0.4 < aspect_ratio < 2.5:
                if x > 5 and y > 5 and (x + w) < (w_img - 5) and (y + h) < (h_img - 5):
                    bubbles.append((x, y, w, h))

    bubbles = sorted(bubbles, key=lambda b: b[1])
    return bubbles


# --- ML-based bubble detection ------------------------------------
_yolo_model = None
_yolo_load_failed = False


def _get_bubble_model():
    global _yolo_model, _yolo_load_failed
    if _yolo_model is not None or _yolo_load_failed:
        return _yolo_model
    try:
        from ultralytics import YOLO
        model_path = os.getenv("BUBBLE_MODEL_PATH", "models/comic-speech-bubble-detector.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Bubble model weights not found at {model_path}")
        _yolo_model = YOLO(model_path)
    except Exception as e:
        print(f"[bubble-detector] ML model unavailable: {e}")
        _yolo_load_failed = True
        _yolo_model = None
    return _yolo_model


def detect_bubbles_ml(image_path, conf=0.25):
    model = _get_bubble_model()
    if model is None:
        raise RuntimeError("ML bubble detector not loaded")
    results = model.predict(image_path, conf=conf, verbose=False)
    bubbles = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        w, h = x2 - x1, y2 - y1
        if w > 0 and h > 0:
            bubbles.append((x1, y1, w, h))
    bubbles = sorted(bubbles, key=lambda b: b[1])
    return bubbles


def detect_bubbles(image_path):
    """Try ML first, fallback to OpenCV."""
    try:
        return detect_bubbles_ml(image_path)
    except Exception as e:
        print(f"[bubble-detector] Falling back to OpenCV: {e}")
        return detect_bubbles_opencv(image_path)


# --- OCR Loader ----------------------------------------------------
_ocr_reader = None


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['ja', 'en'], gpu=False, model_storage_directory="models")
    return _ocr_reader


def merge_boxes(boxes_with_text, y_threshold=30, x_threshold=50):
    """
    Groups vertically and horizontally close text bounding boxes.
    """
    if not boxes_with_text:
        return []
    
    # Sort primarily by top coordinate (y)
    sorted_items = sorted(boxes_with_text, key=lambda item: item["box"][1])
    
    merged = []
    for item in sorted_items:
        bx, by, bw, bh = item["box"]
        b_text = item["text"]
        
        merged_any = False
        for m in merged:
            mx, my, mw, mh = m["box"]
            
            # Check if vertical gap is small
            vertical_close = (by - (my + mh)) < y_threshold and (by >= my - 5)
            
            # Check if they overlap or are very close horizontally
            horizontal_close = not (bx + bw + x_threshold < mx or mx + mw + x_threshold < bx)
            
            if vertical_close and horizontal_close:
                new_x = min(mx, bx)
                new_y = min(my, by)
                new_w = max(mx + mw, bx + bw) - new_x
                new_h = max(my + mh, by + bh) - new_y
                
                m["box"] = (new_x, new_y, new_w, new_h)
                m["text"] = m["text"] + " " + b_text
                merged_any = True
                break
                
        if not merged_any:
            merged.append({"box": (bx, by, bw, bh), "text": b_text})
            
    return merged


def get_page_dialogues(image_path):
    """
    Detects actual dialogues on a page.
    Combines YOLO/OpenCV bubble detection with EasyOCR text detection.
    Only keeps regions that contain actual text, and maps them to clean boundaries.
    """
    try:
        # 1. Run OCR on the whole image
        reader = None
        ocr_results = []
        try:
            reader = get_ocr_reader()
            ocr_results = reader.readtext(image_path)
        except Exception as e:
            print(f"[OCR-detect] Failed to run EasyOCR: {e}")

        ocr_boxes = []
        for bbox, text, conf in ocr_results:
            text = text.strip()
            if not text:
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x = int(min(xs))
            y = int(min(ys))
            w = int(max(xs) - x)
            h = int(max(ys) - y)
            if w > 0 and h > 0:
                ocr_boxes.append({"box": (x, y, w, h), "text": text})

        grouped_ocr = merge_boxes(ocr_boxes)

        # 2. Run Bubble Detection
        bubbles = []
        try:
            bubbles = detect_bubbles(image_path)
        except Exception as e:
            print(f"[Bubble-detect] Failed to detect bubbles: {e}")

        # 3. Match grouped OCR blocks with bubbles
        dialogues = []
        matched_ocr_indices = set()

        for bx, by, bw, bh in bubbles:
            bubble_text_parts = []
            bubble_contains_text = False
            
            for ocr_idx, ocr_item in enumerate(grouped_ocr):
                ox, oy, ow, oh = ocr_item["box"]
                
                # Overlap check
                ix = max(bx, ox)
                iy = max(by, oy)
                iw = min(bx + bw, ox + ow) - ix
                ih = min(by + bh, oy + oh) - iy
                
                if iw > 0 and ih > 0:
                    overlap_area = iw * ih
                    ocr_area = ow * oh
                    # If 30%+ of the text is inside the bubble
                    if overlap_area / ocr_area > 0.3:
                        bubble_text_parts.append(ocr_item["text"])
                        bubble_contains_text = True
                        matched_ocr_indices.add(ocr_idx)

            if bubble_contains_text:
                combined_text = " ".join(bubble_text_parts).strip()
                dialogues.append({
                    "bbox": [bx, by, bx + bw, by + bh],
                    "text": combined_text
                })

        # 4. Add remaining unmatched OCR text blocks (e.g. orange boxes, colored text)
        for ocr_idx, ocr_item in enumerate(grouped_ocr):
            if ocr_idx not in matched_ocr_indices:
                ox, oy, ow, oh = ocr_item["box"]
                pad = 6
                img_w, img_h = 10000, 10000
                try:
                    with Image.open(image_path) as img_temp:
                        img_w, img_h = img_temp.size
                except Exception:
                    pass
                
                x1 = max(0, ox - pad)
                y1 = max(0, oy - pad)
                x2 = min(img_w, ox + ow + pad)
                y2 = min(img_h, oy + oh + pad)
                
                dialogues.append({
                    "bbox": [x1, y1, x2, y2],
                    "text": ocr_item["text"]
                })

        dialogues.sort(key=lambda d: d["bbox"][1])
        return dialogues

    except Exception as e:
        print(f"[Hybrid-dialogue-detection] Error: {e}, falling back to default OpenCV bubble detection")
        bubbles = []
        try:
            bubbles = detect_bubbles(image_path)
        except Exception:
            bubbles = detect_bubbles_opencv(image_path)
        
        fallback_dialogues = []
        for bx, by, bw, bh in bubbles:
            fallback_dialogues.append({
                "bbox": [bx, by, bx + bw, by + bh],
                "text": "[Translate]"
            })
        return fallback_dialogues


def split_text_to_lines(text, max_width, draw, font):
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []
    if current_line:
        lines.append(' '.join(current_line))
    return lines


def _load_font(font_path=None, size=22):
    """Load a TrueType font if available, else fallback to default."""
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def process_phase1_engine(input_path, output_dir, mode_label="normal", progress_cb=None, font_path=None):
    temp_dir = os.path.join(output_dir, "temp_extracted")
    clean_dir = os.path.join(output_dir, "clean_pages")
    numbered_dir = os.path.join(output_dir, "numbered_pages")

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(numbered_dir, exist_ok=True)

    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".pdf":
        images = convert_from_path(input_path, dpi=150)
        for idx, img in enumerate(images):
            img.save(os.path.join(temp_dir, f"{idx + 1:04d}.png"), "PNG")
    elif ext == ".zip":
        with zipfile.ZipFile(input_path, 'r') as z:
            z.extractall(temp_dir)
    elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
        shutil.copy(input_path, os.path.join(temp_dir, "0001" + ext))
    else:
        raise ValueError(f"Unsupported input file type: {ext}")

    ordered_paths = _gather_pages_recursive(temp_dir)
    if not ordered_paths:
        raise Exception("No valid manga pages found inside the uploaded file.")

    translation_map = {}
    txt_lines = [f"# Mode: {mode_label.upper()}", "# Translate text after the '=Dialogue=' marker. Keep the ID matching."]
    total_pages = len(ordered_paths)

    font_num = _load_font(font_path, size=20)

    for page_idx, src_path in enumerate(ordered_paths):
        page_ext = os.path.splitext(src_path)[1].lower() or ".png"
        page_file = f"{page_idx + 1:04d}{page_ext}"

        dialogues = get_page_dialogues(src_path)

        img_clean = Image.open(src_path).convert("RGB")
        img_numbered = img_clean.copy()

        draw_clean = ImageDraw.Draw(img_clean)
        draw_num = ImageDraw.Draw(img_numbered)

        page_key = f"Page{page_idx + 1:03d}"
        translation_map[page_key] = []

        for b_idx, d_item in enumerate(dialogues):
            bubble_id = f"{page_key}_Bubble{b_idx + 1}"
            x1, y1, x2, y2 = d_item["bbox"]
            ocr_text = d_item["text"]

            draw_clean.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))

            draw_num.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            draw_num.ellipse([x1 - 12, y1 - 12, x1 + 12, y1 + 12], fill=(255, 0, 0))
            draw_num.text((x1 - 6, y1 - 9), str(b_idx + 1), fill=(255, 255, 255), font=font_num)

            translation_map[page_key].append({
                "id": bubble_id,
                "bbox": [x1, y1, x2, y2]
            })
            txt_lines.append(f"{bubble_id}=Dialogue={ocr_text}")

        img_clean.save(os.path.join(clean_dir, page_file))
        img_numbered.save(os.path.join(numbered_dir, page_file))

        if progress_cb:
            try:
                progress_cb(page_idx + 1, total_pages)
            except Exception:
                pass

    backup_zip = os.path.join(output_dir, "Manga_Backup.zip")
    map_json_path = os.path.join(output_dir, "translation_map.json")
    with open(map_json_path, 'w') as jf:
        json.dump(translation_map, jf, indent=2)

    with zipfile.ZipFile(backup_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(map_json_path, "translation_map.json")
        for root, _, files in os.walk(clean_dir):
            for file in files:
                z.write(os.path.join(root, file), os.path.join("clean_pages", file))
        for root, _, files in os.walk(numbered_dir):
            for file in files:
                z.write(os.path.join(root, file), os.path.join("numbered_pages", file))

    txt_path = os.path.join(output_dir, "translate_me.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    return backup_zip, txt_path, clean_dir, translation_map


def render_translations(clean_pages_dir, translation_map, txt_path, output_dir, progress_cb=None, font_path=None):
    final_dir = os.path.join(output_dir, "final_rendered_pages")
    os.makedirs(final_dir, exist_ok=True)

    translations = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=Dialogue=' not in line:
                continue
            pid, text = line.split('=Dialogue=', 1)
            translations[pid.strip()] = text.strip()

    font = _load_font(font_path, size=22)
    page_files = sorted(os.listdir(clean_pages_dir), key=_natural_key)

    rendered_count = 0
    total_map_pages = len(translation_map)
    for map_idx, (page_key, bubbles) in enumerate(translation_map.items()):
        try:
            p_idx = int(page_key.replace("Page", "")) - 1
            page_file = page_files[p_idx]
        except (ValueError, IndexError):
            continue

        img_path = os.path.join(clean_pages_dir, page_file)
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for b in bubbles:
            text = translations.get(b["id"])
            if not text or text == "[Translate]":
                continue
            x1, y1, x2, y2 = b["bbox"]
            w_box = max(x2 - x1 - 10, 20)
            h_box = y2 - y1

            lines = split_text_to_lines(text, w_box, draw, font)
            line_bbox = draw.textbbox((0, 0), "Ag", font=font)
            line_h = (line_bbox[3] - line_bbox[1]) + 6
            total_h = len(lines) * line_h
            curr_y = y1 + max((h_box - total_h) // 2, 0)

            for line in lines:
                lb = draw.textbbox((0, 0), line, font=font)
                line_w = lb[2] - lb[0]
                curr_x = x1 + max((w_box - line_w) // 2, 0)
                draw.text((curr_x, curr_y), line, fill=(0, 0, 0), font=font)
                curr_y += line_h

        img.save(os.path.join(final_dir, page_file))
        rendered_count += 1

        if progress_cb:
            try:
                progress_cb(map_idx + 1, total_map_pages)
            except Exception:
                pass

    final_zip = os.path.join(output_dir, "Final_Manga_Translated.zip")
    with zipfile.ZipFile(final_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(final_dir), key=_natural_key):
            z.write(os.path.join(final_dir, f), f)

    return final_zip, rendered_count


def process_phase2_engine(backup_zip, txt_path, output_dir, progress_cb=None, font_path=None):
    """Used by /repeat: user re-uploaded the backup zip, so we extract it fresh."""
    extract_temp = os.path.join(output_dir, "backup_extracted")
    os.makedirs(extract_temp, exist_ok=True)

    with zipfile.ZipFile(backup_zip, 'r') as z:
        z.extractall(extract_temp)

    map_path = os.path.join(extract_temp, "translation_map.json")
    if not os.path.exists(map_path):
        raise FileNotFoundError("Invalid Backup ZIP — translation_map.json missing.")
    with open(map_path, 'r') as f:
        translation_map = json.load(f)

    clean_pages_dir = os.path.join(extract_temp, "clean_pages")
    if not os.path.isdir(clean_pages_dir):
        raise FileNotFoundError("Invalid Backup ZIP — clean_pages/ folder missing.")

    final_zip, count = render_translations(clean_pages_dir, translation_map, txt_path, output_dir, progress_cb=progress_cb, font_path=font_path)
    if count == 0:
        raise Exception("No dialogues could be rendered. Check your .txt file format (IDs must match).")
    return final_zip


def process_phase2_from_local(clean_pages_dir, translation_map, txt_path, output_dir, progress_cb=None, font_path=None):
    """Used by the immediate (same-run) flow: no zip download needed at all."""
    final_zip, count = render_translations(clean_pages_dir, translation_map, txt_path, output_dir, progress_cb=progress_cb, font_path=font_path)
    if count == 0:
        raise Exception("No dialogues could be rendered. Check your .txt file format (IDs must match).")
    return final_zip
