import os
import json
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pdf2image import convert_from_path
import easyocr
from tqdm import tqdm


PDF_PATH = "AI.pdf"
CACHE_DIR = Path("ocr_cache_easyocr")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SYMBOL_PATTERNS = [
    (r"↔", "biconditional_symbol"),
    (r"→", "implication_symbol"),
    (r"⊨", "entails_symbol"),
    (r"¬", "not_symbol"),
    (r"∧|Ù", "and_symbol"),
    (r"∨|Ú", "or_symbol"),
    (r"∞", "infinity_symbol"),
    (r"∑", "sum_symbol"),
    (r"√", "sqrt_symbol"),
    (r"μ", "mu_symbol"),
    (r"σ", "sigma_symbol"),
    (r"softmax", "softmax"),
    (r"argmax", "argmax"),
    (r"maxsim", "maxsim"),
]

ANNOTATION_SCHEMA = {
    "page_title": "",
    "visual_type": [],
    "main_entities": [],
    "visible_objects": [],
    "salient_numbers": [],
    "source_venue_year": [],
    "diagram_relations": [],
    "chart_axes_or_legend": [],
    "visual_markers": [],
    "key_grounded_facts": [],
}

VENUE_TERMS = [
    "neurips", "nips", "iclr", "cvpr", "naacl", "aaai", "jmlr", "tmlr", "uist",
    "eccv", "icme", "ijcai", "plmr", "arxiv", "icml", "icra", "tacl",
]

ROLE_PRIORITY = {"title": 0, "caption": 1, "source": 2, "equation": 3, "legend": 4, "body": 5}



def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def make_cache_path(pdf_path: str) -> Path:
    pdf_name = Path(pdf_path).stem
    file_hash = sha1_of_file(pdf_path)[:12]
    return CACHE_DIR / f"{pdf_name}_{file_hash}.json"


def load_cache(cache_path: Path) -> Dict[str, Any]:
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_cache(cache_path: Path, data: Dict[str, Any]) -> None:
    tmp_path = cache_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)
    tmp_path.replace(cache_path)


def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _bbox_xyxy(bbox: Any) -> Optional[List[float]]:
    try:
        pts = np.asarray(bbox, dtype=float)
        if pts.size < 8:
            return None
        xs = pts[:, 0]
        ys = pts[:, 1]
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    except Exception:
        return None


def _region_label(cx: float, cy: float) -> str:
    if cy < 0.33:
        row = "top"
    elif cy < 0.67:
        row = "middle"
    else:
        row = "bottom"

    if cx < 0.33:
        col = "left"
    elif cx < 0.67:
        col = "center"
    else:
        col = "right"
    return f"{row}_{col}"


def classify_block_role(block: Dict[str, Any], fallback_title: bool = False) -> str:
    text = normalize_text(block.get("text", ""))
    low = text.lower()
    region = str(block.get("region", "unknown"))
    n_words = len(low.split())
    if not low:
        return "body"
    if low.startswith(("figure", "fig.", "table", "tab.")) or (region.startswith("bottom") and n_words >= 6 and len(low) >= 30):
        return "caption"
    if "source:" in low or "http" in low or "www." in low or any(v in low for v in VENUE_TERMS) or "@" in low:
        return "source"
    if any(sym in low for sym in ["softmax", "argmax", "maxsim"]) or any(ch in text for ch in ["=", "→", "↔", "⊨", "∑", "√", "μ", "σ"]):
        return "equation"
    if ("legend" in low or "axis" in low or low in {"x", "y"}) and n_words <= 8:
        return "legend"
    if fallback_title:
        return "title"
    if region.startswith("top") and 2 <= n_words <= 14 and len(low) >= 8 and len(low) <= 120 and not re.fullmatch(r"[0-9 .:/%-]+", low):
        return "title"
    return "body"


def assign_block_roles(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not blocks:
        return blocks
    out = [dict(b) for b in blocks]
    top_candidates = []
    for idx, b in enumerate(out):
        region = str(b.get("region", "unknown"))
        text = normalize_text(b.get("text", ""))
        if region.startswith("top") and 2 <= len(text.split()) <= 14 and len(text) >= 8:
            top_candidates.append((idx, len(text)))
    top_candidates = sorted(top_candidates, key=lambda x: -x[1])[:2]
    fallback_title_idx = {idx for idx, _ in top_candidates}
    for idx, b in enumerate(out):
        b["role"] = classify_block_role(b, fallback_title=(idx in fallback_title_idx and idx == top_candidates[0][0] if top_candidates else False))
    return out



def enrich_block_geometry(blocks: List[Dict[str, Any]], image_w: int, image_h: int) -> List[Dict[str, Any]]:
    enriched = []
    for block in blocks:
        new_block = dict(block)
        xyxy = _bbox_xyxy(block.get("bbox"))
        if xyxy is not None and image_w > 0 and image_h > 0:
            x0, y0, x1, y1 = xyxy
            cx = (x0 + x1) / 2.0 / image_w
            cy = (y0 + y1) / 2.0 / image_h
            width = max(0.0, x1 - x0) / image_w
            height = max(0.0, y1 - y0) / image_h
            new_block.update({
                "bbox_xyxy": [x0, y0, x1, y1],
                "center": [cx, cy],
                "size": [width, height],
                "region": _region_label(cx, cy),
            })
        else:
            new_block.update({
                "bbox_xyxy": None,
                "center": None,
                "size": None,
                "region": "unknown",
            })
        enriched.append(new_block)
    return enriched


def sort_blocks_reading_order(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_fn(b: Dict[str, Any]):
        xyxy = b.get("bbox_xyxy")
        role = str(b.get("role", "body"))
        pr = ROLE_PRIORITY.get(role, 99)
        if xyxy is None:
            return (10**9, pr, 10**9)
        x0, y0, _, _ = xyxy
        return (round(y0 / 12.0) * 12.0, pr, x0)

    return sorted(blocks, key=key_fn)


def normalize_easyocr_result(result: Any) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if not result:
        return blocks

    for item in result:
        if not item or len(item) < 3:
            continue

        bbox, text, confidence = item[0], item[1], item[2]
        text = "" if text is None else str(text).strip()
        if not text:
            continue

        blocks.append({
            "text": normalize_text(text),
            "confidence": float(confidence) if confidence is not None else None,
            "bbox": to_jsonable(bbox),
        })
    return blocks


def build_page_text(blocks: List[Dict[str, Any]]) -> str:
    return "\n".join(block["text"] for block in blocks if block["text"].strip())


def extract_page_signals(text: str, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", text)))
    percents = sorted(set(re.findall(r"\b\d+(?:\.\d+)?%\b", text)))
    money = sorted(set(re.findall(r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|trillion|m|b|k))?", text, flags=re.I)))
    numbers = sorted(set(re.findall(r"\b\d+(?:\.\d+)?(?:%|x|b|m|k|t)?\b", text, flags=re.I)))

    symbols: List[str] = []
    for pat, label in SYMBOL_PATTERNS:
        if re.search(pat, text, flags=re.I):
            symbols.append(label)
    symbols = sorted(set(symbols))

    source_lines = []
    caption_lines = []
    title_lines = []
    equation_lines = []
    role_texts: Dict[str, List[str]] = {}
    region_role_texts: Dict[str, List[str]] = {}
    for b in blocks:
        line = normalize_text(b.get("text", ""))
        ll = line.lower().strip()
        if not ll:
            continue
        role = str(b.get("role", "body"))
        role_texts.setdefault(role, []).append(line)
        rr = f"{b.get('region', 'unknown')}::{role}"
        region_role_texts.setdefault(rr, []).append(line)
        if role == "source" or "source:" in ll or re.search(r"\b(" + "|".join(VENUE_TERMS) + r")\b", ll):
            source_lines.append(line)
        if role == "caption" or re.search(r"^(figure|fig\.|table|tab\.)", ll):
            caption_lines.append(line)
        if role == "title":
            title_lines.append(line)
        if role == "equation":
            equation_lines.append(line)

    region_counts: Dict[str, int] = {}
    for b in blocks:
        region = b.get("region", "unknown")
        region_counts[region] = region_counts.get(region, 0) + 1

    return {
        "years": years,
        "percents": percents,
        "money": money,
        "numbers": numbers,
        "symbols": symbols,
        "source_lines": source_lines[:12],
        "caption_lines": caption_lines[:12],
        "title_lines": title_lines[:6],
        "equation_lines": equation_lines[:12],
        "role_texts": {k: v[:20] for k, v in role_texts.items()},
        "region_role_texts": {k: v[:16] for k, v in region_role_texts.items()},
        "region_counts": region_counts,
    }


def is_visual_page_candidate(full_text: str, blocks: List[Dict[str, Any]]) -> bool:
    text_len = len(full_text)
    block_count = len(blocks)
    has_figure_term = bool(re.search(r"\b(figure|fig\.|table|chart|diagram|plot|qualitative|comparison|results)\b", full_text, flags=re.I))
    has_large_bottom_caption = any(
        (b.get("region") or "").startswith("bottom") and len(b.get("text", "")) >= 30
        for b in blocks
    )
    sparse_text_layout = text_len < 500 and block_count < 22
    return bool(has_figure_term or has_large_bottom_caption or sparse_text_layout)


class StructuredPageAnnotator:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        device_map: str = "auto",
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.model = None
        self.processor = None
        if not enabled:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[WARN] Failed to load structured annotator {model_name}: {e}")
            self.enabled = False
            self.model = None
            self.processor = None

    def _prompt(self, ocr_text: str) -> str:
        return (
            "You are grounding a lecture slide into structured JSON. "
            "Describe only directly visible content. Do not infer hidden meaning. "
            "Do not add facts not shown on the slide. Output valid JSON only with keys: "
            "page_title, visual_type, main_entities, visible_objects, salient_numbers, "
            "source_venue_year, diagram_relations, chart_axes_or_legend, visual_markers, key_grounded_facts.\n\n"
            "Guidelines:\n"
            "- page_title: short visible title if any\n"
            "- visual_type: choose from diagram, chart, table, example_image, screenshot, equation, architecture, qualitative_results, text_only\n"
            "- main_entities: named models, methods, datasets, modules, venues visibly mentioned\n"
            "- visible_objects: concrete pictured objects or screenshot elements\n"
            "- salient_numbers: visible values, years, percentages, money, counts\n"
            "- source_venue_year: visible citation snippets\n"
            "- diagram_relations: short grounded relations like 'planner sends action to retriever'\n"
            "- chart_axes_or_legend: axis or legend terms visible on charts\n"
            "- visual_markers: red box, arrow, highlighted region, colored legend, etc\n"
            "- key_grounded_facts: 1-5 short factual statements copied or closely paraphrased from visible content\n\n"
            f"OCR text for context (may be noisy): {ocr_text[:1800]}"
        )

    def _safe_parse(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        parsed = None
        try:
            parsed = json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None
        if not isinstance(parsed, dict):
            return dict(ANNOTATION_SCHEMA)
        out = dict(ANNOTATION_SCHEMA)
        for k in out.keys():
            v = parsed.get(k, out[k])
            if isinstance(out[k], list):
                out[k] = [str(x).strip() for x in v[:12]] if isinstance(v, list) else []
            else:
                out[k] = str(v).strip() if v is not None else ""
        return out

    def annotate(self, image, ocr_text: str) -> Dict[str, Any]:
        if not self.enabled or self.model is None or self.processor is None:
            return dict(ANNOTATION_SCHEMA)
        try:
            import torch

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self._prompt(ocr_text)},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            try:
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            except Exception:
                pass
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            gen = generated_ids[0][inputs["input_ids"].shape[1]:]
            out_text = self.processor.decode(gen, skip_special_tokens=True)
            return self._safe_parse(out_text)
        except Exception as e:
            print(f"[WARN] Structured annotation failed: {e}")
            return dict(ANNOTATION_SCHEMA)


def ocr_pdf_to_cache(
    pdf_path: str,
    dpi: int = 200,
    languages: List[str] = None,
    use_gpu: bool = True,
    annotate_visual_pages: bool = False,
    annotate_all_pages: bool = False,
    annotation_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    annotation_max_new_tokens: int = 256,
    overwrite_annotations: bool = False,
) -> Dict[str, Any]:
    """OCR a PDF page-by-page with EasyOCR and persistent cache.

    Optional structured page annotations are grounded, image-based JSON summaries
    intended for later retrieval, not free-form explanations.
    """
    if languages is None:
        languages = ["en"]

    cache_path = make_cache_path(pdf_path)
    cache = load_cache(cache_path)

    if "meta" not in cache:
        cache["meta"] = {
            "pdf_path": str(Path(pdf_path).resolve()),
            "pdf_name": Path(pdf_path).name,
            "file_sha1": sha1_of_file(pdf_path),
            "ocr_engine": "EasyOCR",
            "ocr_lang": languages,
            "version": 3,
            "dpi": dpi,
            "use_gpu": use_gpu,
        }

    cache["meta"].update({
        "annotation_model_name": annotation_model_name if (annotate_visual_pages or annotate_all_pages) else "",
        "annotation_mode": "all" if annotate_all_pages else ("visual_only" if annotate_visual_pages else "off"),
    })

    if "pages" not in cache:
        cache["pages"] = {}

    reader = easyocr.Reader(languages, gpu=use_gpu, verbose=False)
    images = convert_from_path(pdf_path, dpi=dpi)

    annotator = None
    if annotate_visual_pages or annotate_all_pages:
        annotator = StructuredPageAnnotator(
            model_name=annotation_model_name,
            max_new_tokens=annotation_max_new_tokens,
            enabled=True,
        )

    for page_idx, image in tqdm(list(enumerate(images)), desc="OCR pages"):
        key = str(page_idx)
        image_w, image_h = image.size

        page_needs_refresh = key not in cache["pages"]
        if not page_needs_refresh:
            # Still allow later annotation enrichment without re-running OCR.
            if not (annotate_visual_pages or annotate_all_pages):
                continue
            if not overwrite_annotations and cache["pages"][key].get("annotation"):
                continue

        page_record = cache["pages"].get(key, {})
        if key not in cache["pages"]:
            img_np = np.array(image)
            result = reader.readtext(
                img_np,
                detail=1,
                paragraph=False,
            )
            blocks = normalize_easyocr_result(result)
            blocks = enrich_block_geometry(blocks, image_w=image_w, image_h=image_h)
            blocks = assign_block_roles(blocks)
            blocks = sort_blocks_reading_order(blocks)
            full_text = build_page_text(blocks)
            signals = extract_page_signals(full_text, blocks)
            visual_candidate = is_visual_page_candidate(full_text, blocks)

            page_record = {
                "page_index": page_idx,
                "image_size": [image_w, image_h],
                "text": full_text,
                "blocks": blocks,
                "n_blocks": len(blocks),
                "n_chars": len(full_text),
                "visual_candidate": visual_candidate,
                **signals,
            }
            cache["pages"][key] = page_record
            save_cache(cache_path, cache)

        if annotator is not None:
            should_annotate = annotate_all_pages or bool(cache["pages"][key].get("visual_candidate", False))
            if should_annotate and (overwrite_annotations or not cache["pages"][key].get("annotation")):
                annotation = annotator.annotate(image=image, ocr_text=cache["pages"][key].get("text", ""))
                cache["pages"][key]["annotation"] = annotation
                save_cache(cache_path, cache)

    return cache


def load_ocr_texts(pdf_path: str) -> Dict[int, str]:
    cache_path = make_cache_path(pdf_path)
    if not cache_path.exists():
        ocr_pdf_to_cache(pdf_path)

    cache = load_cache(cache_path)
    return {int(k): v["text"] for k, v in cache.get("pages", {}).items()}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_path", default=PDF_PATH)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--languages", nargs="+", default=["en"])
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--annotate_visual_pages", action="store_true")
    ap.add_argument("--annotate_all_pages", action="store_true")
    ap.add_argument("--annotation_model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--annotation_max_new_tokens", type=int, default=256)
    ap.add_argument("--overwrite_annotations", action="store_true")
    args = ap.parse_args()

    cache = ocr_pdf_to_cache(
        pdf_path=args.pdf_path,
        dpi=args.dpi,
        languages=args.languages,
        use_gpu=args.use_gpu,
        annotate_visual_pages=args.annotate_visual_pages,
        annotate_all_pages=args.annotate_all_pages,
        annotation_model_name=args.annotation_model_name,
        annotation_max_new_tokens=args.annotation_max_new_tokens,
        overwrite_annotations=args.overwrite_annotations,
    )
    print("Done.")
    print("Cache file:", make_cache_path(args.pdf_path))

    page_texts = load_ocr_texts(args.pdf_path)
    print("\n=== Page 0 preview ===")
    print(page_texts.get(0, "")[:1000])
