from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Retrieval backbone: prefer the best answer-driven version if available.
try:
    import rag_page_locator_simple_v3_1_answer_patch as backbone_mod
except Exception:
    import AI_2026.hw1._hw1_D14922031_task2.rag_page_locator_simple_v3_structured as backbone_mod


def dumps_compact(x: Any) -> str:
    def _default(v: Any) -> Any:
        try:
            import numpy as _np
            if isinstance(v, _np.integer):
                return int(v)
            if isinstance(v, _np.floating):
                return float(v)
            if isinstance(v, _np.ndarray):
                return v.tolist()
        except Exception:
            pass
        return str(v)

    return json.dumps(x, ensure_ascii=False, default=_default)


@dataclass
class Config:
    pdf_path: str = "AI.pdf"
    question_path: str = "HW1_questions.json"
    output_path: str = "submission.csv"
    debug_path: str = "debug_vlm.csv"
    trace_path: str = "trace_vlm.jsonl"
    cache_path: str = ""

    # inherited backbone knobs
    embed_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    batch_size: int = 16
    page_base: int = 1
    top_k_rerank: int = 6
    rerank_margin: float = 0.05

    # VLM verifier knobs
    vlm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    top_k_candidates: int = 24
    top_k_vlm: int = 8
    max_new_tokens: int = 160
    load_in_4bit: bool = False
    page_dpi: int = 144
    page_max_edge: int = 1280
    image_cache_dir: str = "page_image_cache_vlm"
    use_vlm: bool = True

    # score fusion
    retrieval_weight_text: float = 0.45
    retrieval_weight_special: float = 0.35
    exact_weight_text: float = 0.10
    exact_weight_special: float = 0.15
    vlm_weight_text: float = 0.45
    vlm_weight_special: float = 0.50


class PageImageCache:
    def __init__(self, pdf_path: str, cache_dir: str, dpi: int = 144, max_edge: int = 1280):
        self.pdf_path = pdf_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.max_edge = max_edge
        self.doc = fitz.open(pdf_path)
        stem = Path(pdf_path).stem
        self.pdf_dir = self.cache_dir / stem
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def render_page(self, page_num_1based: int) -> Path:
        out_path = self.pdf_dir / f"page_{page_num_1based:04d}.png"
        if out_path.exists():
            return out_path

        page = self.doc[page_num_1based - 1]
        zoom = self.dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if max(img.size) > self.max_edge:
            img.thumbnail((self.max_edge, self.max_edge))
        img.save(out_path)
        return out_path


class QwenVLVerifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.processor = None
        self._init_model()

    def _init_model(self) -> None:
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        except Exception:
            from transformers import AutoModelForVision2Seq as ModelCls

        model_kwargs: Dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        if self.cfg.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as e:
                raise RuntimeError("--load_in_4bit requires bitsandbytes + recent transformers") from e

        self.processor = AutoProcessor.from_pretrained(self.cfg.vlm_model_name, trust_remote_code=True)
        self.model = ModelCls.from_pretrained(self.cfg.vlm_model_name, **model_kwargs)
        self.model.eval()

    @staticmethod
    def _safe_json_parse(text: str) -> Dict[str, Any]:
        text = text.strip()
        candidates = []
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            candidates.append(m.group(0))
        candidates.append(text)
        for cand in candidates:
            try:
                return json.loads(cand)
            except Exception:
                continue

        # fallback: recover score + verdict heuristically
        score = None
        m_score = re.search(r'"?score"?\s*[:=]\s*(\d{1,3})', text, flags=re.I)
        if m_score:
            score = int(m_score.group(1))
        else:
            m_any = re.search(r"\b(\d{1,3})\b", text)
            if m_any:
                score = int(m_any.group(1))
        score = 50 if score is None else max(0, min(100, score))
        verdict = "yes" if re.search(r"\byes\b|\btrue\b", text, flags=re.I) else "no"
        return {"score": score, "verdict": verdict, "reason": text[:400], "evidence": []}

    def _build_prompt(self, question: str, query_type: str, candidate: Dict[str, Any]) -> str:
        title_hint = str(candidate.get("title_hint", ""))
        source_text = str(candidate.get("source_text", ""))
        symbol_text = str(candidate.get("symbol_text", ""))
        annotation_text = str(candidate.get("annotation_text", ""))
        top_blocks = candidate.get("top_ocr_blocks", []) or []
        block_text = " | ".join(str(b.get("text", "")) for b in top_blocks[:3])

        type_instructions = {
            "citation": "Prioritize exact paper title, venue, journal, year, and source lines.",
            "formula": "Prioritize exact formulas, symbols, variable letters, equations, and mathematical notation.",
            "location": "Prioritize title, section heading, page layout, captions, legends, and whether this page is the place being referred to.",
            "visual": "Prioritize visual objects, red boxes, arrows, figures, charts, tables, captions, and diagram relations.",
            "literal": "Prioritize exact numbers, percentages, money values, years, and table/chart evidence.",
            "text": "Prioritize whether this page directly states the answer, not just related topic overlap.",
        }
        instruction = type_instructions.get(query_type, type_instructions["text"])

        return (
            "You are verifying whether a single lecture slide page is the BEST answer page for a question. "
            "Use the PAGE IMAGE as primary evidence, and use OCR/text as secondary evidence. "
            "Be strict: related topic is not enough; the page should directly contain the answer or the strongest supporting evidence.\n\n"
            f"Question: {question}\n"
            f"Query type: {query_type}\n"
            f"Verification focus: {instruction}\n\n"
            f"Page title hint: {title_hint}\n"
            f"Source text: {source_text}\n"
            f"Symbol text: {symbol_text}\n"
            f"Annotation text: {annotation_text}\n"
            f"Top OCR blocks: {block_text}\n\n"
            "Return STRICT JSON only with this schema:\n"
            '{"score": 0-100, "verdict": "yes" or "no", "page_kind": "short label", '
            '"evidence": ["short evidence 1", "short evidence 2"], "reason": "short reason"}'
        )

    def verify(self, image_path: Path, question: str, query_type: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        prompt = self._build_prompt(question, query_type, candidate)
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        # decode only generated tail
        prompt_len = inputs["input_ids"].shape[1]
        tail = output_ids[:, prompt_len:]
        text_out = self.processor.batch_decode(tail, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        parsed = self._safe_json_parse(text_out)
        parsed["raw"] = text_out[:1000]
        parsed["score"] = float(max(0.0, min(100.0, float(parsed.get("score", 0.0)))))
        return parsed


class VLMPageRetriever:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base_cfg = self._make_base_cfg(cfg)
        self.backbone = backbone_mod.StructuredSimpleRAGPageLocator(self.base_cfg)
        self.renderer = PageImageCache(cfg.pdf_path, cfg.image_cache_dir, cfg.page_dpi, cfg.page_max_edge)
        self.verifier = QwenVLVerifier(cfg) if cfg.use_vlm else None

    @staticmethod
    def _make_base_cfg(cfg: Config):
        base_field_names = {f.name for f in fields(backbone_mod.Config)}
        kwargs = {k: v for k, v in asdict(cfg).items() if k in base_field_names}
        base_cfg = backbone_mod.Config(**kwargs)
        # Important: backbone.predict_one() truncates returned candidates to top_k_debug.
        # Raise it here so the VLM verifier can actually see more than the old debug ceiling.
        base_cfg.top_k_debug = max(
            int(getattr(base_cfg, "top_k_debug", 10)),
            int(cfg.top_k_candidates),
            int(cfg.top_k_vlm),
            24,
        )
        return base_cfg

    @staticmethod
    def _normalize_retrieval_scores(cands: Sequence[Dict[str, Any]]) -> Dict[int, float]:
        if not cands:
            return {}
        vals = np.asarray([float(c.get("score", 0.0)) for c in cands], dtype=np.float32)
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-8:
            return {int(c["page"]): 1.0 for c in cands}
        return {int(c["page"]): float((float(c["score"]) - lo) / (hi - lo)) for c in cands}

    @staticmethod
    def _exact_evidence_score(c: Dict[str, Any], qtype: str) -> float:
        score = 0.0
        score += 0.30 * float(c.get("source_bonus", 0.0))
        score += 0.25 * float(c.get("number_bonus", 0.0))
        score += 0.25 * float(c.get("symbol_bonus", 0.0))
        score += 0.10 * float(c.get("title_bonus", 0.0))
        score += 0.10 * float(c.get("best_block_sim", 0.0))
        if qtype in {"visual", "location"}:
            score += 0.10 * float(c.get("spatial_bonus", 0.0))
            score += 0.10 * float(c.get("annotation_bonus", 0.0))
        return float(max(0.0, min(1.0, score)))

    def _fuse_scores(self, retrieval_norm: float, exact_score: float, vlm_score: float, qtype: str) -> float:
        if qtype == "text":
            return (
                self.cfg.retrieval_weight_text * retrieval_norm
                + self.cfg.exact_weight_text * exact_score
                + self.cfg.vlm_weight_text * vlm_score
            )
        return (
            self.cfg.retrieval_weight_special * retrieval_norm
            + self.cfg.exact_weight_special * exact_score
            + self.cfg.vlm_weight_special * vlm_score
        )

    def predict_one(self, question: str) -> Dict[str, Any]:
        base = self.backbone.predict_one(question)
        qtype = str(base.get("query_type", "text"))
        all_cands = list(base.get("top_candidates", []))
        if len(all_cands) < self.cfg.top_k_candidates:
            # backbone top_candidates may be truncated by debug limit; recompute from pages if needed is not trivial.
            # In that case, just use available candidates.
            pass
        top_cands = all_cands[: self.cfg.top_k_candidates]
        retrieval_norm = self._normalize_retrieval_scores(top_cands)

        # VLM rerank only top_k_vlm pages.
        vlm_subset = top_cands[: self.cfg.top_k_vlm]
        fused_rows: List[Dict[str, Any]] = []
        vlm_debug: List[Dict[str, Any]] = []

        if self.verifier is None:
            best_page = int(top_cands[0]["page"])
            return {
                **base,
                "pred_page": best_page if self.cfg.page_base == 1 else best_page - 1,
                "vlm_candidates": [],
                "final_candidates": top_cands,
            }

        for c in vlm_subset:
            page_num = int(c["page"])
            image_path = self.renderer.render_page(page_num)
            judge = self.verifier.verify(image_path, question, qtype, c)
            vlm_score = float(judge.get("score", 0.0)) / 100.0
            exact_score = self._exact_evidence_score(c, qtype)
            fused = self._fuse_scores(retrieval_norm.get(page_num, 0.0), exact_score, vlm_score, qtype)
            row = dict(c)
            row["retrieval_norm"] = retrieval_norm.get(page_num, 0.0)
            row["exact_score"] = exact_score
            row["vlm_score"] = vlm_score
            row["vlm_verdict"] = judge.get("verdict", "")
            row["vlm_reason"] = str(judge.get("reason", ""))[:300]
            row["vlm_evidence"] = judge.get("evidence", [])
            row["vlm_page_kind"] = judge.get("page_kind", "")
            row["fused_score"] = fused
            fused_rows.append(row)
            vlm_debug.append({
                "page": page_num,
                "retrieval_norm": retrieval_norm.get(page_num, 0.0),
                "exact_score": exact_score,
                "vlm_score": vlm_score,
                "fused_score": fused,
                "judge": judge,
            })

        # keep non-VLM candidates behind VLM-evaluated ones using retrieval score only
        evaluated_pages = {int(r["page"]) for r in fused_rows}
        for c in top_cands:
            page_num = int(c["page"])
            if page_num in evaluated_pages:
                continue
            exact_score = self._exact_evidence_score(c, qtype)
            fused = self._fuse_scores(retrieval_norm.get(page_num, 0.0), exact_score, 0.0, qtype)
            row = dict(c)
            row["retrieval_norm"] = retrieval_norm.get(page_num, 0.0)
            row["exact_score"] = exact_score
            row["vlm_score"] = 0.0
            row["vlm_verdict"] = "not_checked"
            row["vlm_reason"] = ""
            row["vlm_evidence"] = []
            row["vlm_page_kind"] = ""
            row["fused_score"] = fused
            fused_rows.append(row)

        fused_rows.sort(key=lambda x: (-float(x["fused_score"]), -float(x.get("score", 0.0))))
        pred_page = int(fused_rows[0]["page"])
        pred_out = pred_page if self.cfg.page_base == 1 else pred_page - 1

        return {
            "question": question,
            "query_type": qtype,
            "normalized_query": base.get("normalized_query", ""),
            "query_variants": base.get("query_variants", []),
            "query_keywords": base.get("query_keywords", []),
            "pred_page": pred_out,
            "top_candidates": base.get("top_candidates", []),
            "final_candidates": fused_rows[: self.base_cfg.top_k_debug],
            "vlm_candidates": vlm_debug,
            "effective_backbone_top_k": int(self.base_cfg.top_k_debug),
            "rerank_info": base.get("rerank_info", {}),
        }

    def run(self, questions: Sequence[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
        submission_rows: List[Dict[str, Any]] = []
        debug_rows: List[Dict[str, Any]] = []
        traces: List[Dict[str, Any]] = []

        for item in tqdm(questions, desc="Predicting pages with VLM verifier"):
            qid = int(item["ID"])
            out = self.predict_one(str(item["question"]))
            submission_rows.append({"ID": qid, "TARGET": int(out["pred_page"])})
            debug_rows.append({
                "ID": qid,
                "question": out["question"],
                "query_type": out["query_type"],
                "normalized_query": out["normalized_query"],
                "query_variants": dumps_compact(out["query_variants"]),
                "query_keywords": dumps_compact(out["query_keywords"]),
                "pred_page": int(out["pred_page"]),
                "effective_backbone_top_k": int(out.get("effective_backbone_top_k", 0)),
                "top_candidates": dumps_compact(out["top_candidates"]),
                "final_candidates": dumps_compact(out["final_candidates"]),
                "vlm_candidates": dumps_compact(out["vlm_candidates"]),
            })
            traces.append({
                "ID": qid,
                "question": out["question"],
                "query_type": out["query_type"],
                "normalized_query": out["normalized_query"],
                "query_variants": out["query_variants"],
                "query_keywords": out["query_keywords"],
                "pred_page": int(out["pred_page"]),
                "effective_backbone_top_k": int(out.get("effective_backbone_top_k", 0)),
                "top_candidates": out["top_candidates"],
                "final_candidates": out["final_candidates"],
                "vlm_candidates": out["vlm_candidates"],
            })

        return pd.DataFrame(submission_rows), pd.DataFrame(debug_rows), traces


def write_jsonl(records: Sequence[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(dumps_compact(rec) + "\n")


def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_path", default="AI.pdf")
    ap.add_argument("--question_path", default="HW1_questions.json")
    ap.add_argument("--output_path", default="submission.csv")
    ap.add_argument("--debug_path", default="debug_vlm.csv")
    ap.add_argument("--trace_path", default="trace_vlm.jsonl")
    ap.add_argument("--cache_path", default="")

    ap.add_argument("--embed_model_name", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--page_base", type=int, default=1)
    ap.add_argument("--top_k_rerank", type=int, default=6)
    ap.add_argument("--rerank_margin", type=float, default=0.05)

    ap.add_argument("--vlm_model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--top_k_candidates", type=int, default=24)
    ap.add_argument("--top_k_vlm", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--page_dpi", type=int, default=144)
    ap.add_argument("--page_max_edge", type=int, default=1280)
    ap.add_argument("--image_cache_dir", default="page_image_cache_vlm")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--no_use_vlm", action="store_true")

    args = ap.parse_args()
    return Config(
        pdf_path=args.pdf_path,
        question_path=args.question_path,
        output_path=args.output_path,
        debug_path=args.debug_path,
        trace_path=args.trace_path,
        cache_path=args.cache_path,
        embed_model_name=args.embed_model_name,
        batch_size=args.batch_size,
        page_base=args.page_base,
        top_k_rerank=args.top_k_rerank,
        rerank_margin=args.rerank_margin,
        vlm_model_name=args.vlm_model_name,
        top_k_candidates=args.top_k_candidates,
        top_k_vlm=args.top_k_vlm,
        max_new_tokens=args.max_new_tokens,
        page_dpi=args.page_dpi,
        page_max_edge=args.page_max_edge,
        image_cache_dir=args.image_cache_dir,
        load_in_4bit=args.load_in_4bit,
        use_vlm=not args.no_use_vlm,
    )


def main() -> None:
    cfg = parse_args()
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    questions = backbone_mod.load_questions(cfg.question_path)
    locator = VLMPageRetriever(cfg)
    sub_df, debug_df, traces = locator.run(questions)

    sub_df.to_csv(cfg.output_path, index=False)
    debug_df.to_csv(cfg.debug_path, index=False)
    write_jsonl(traces, cfg.trace_path)

    print(f"Saved submission to: {cfg.output_path}")
    print(f"Saved debug CSV to: {cfg.debug_path}")
    print(f"Saved trace JSONL to: {cfg.trace_path}")
    print(sub_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
