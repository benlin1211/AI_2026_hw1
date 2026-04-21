
import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from PIL import Image
import torch

from diffusers import AutoPipelineForText2Image
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
    HAS_QWEN3_NATIVE = True
except Exception:
    from transformers import AutoModelForImageTextToText
    HAS_QWEN3_NATIVE = False


PROMPT_STYLES = {
    "layout_strict": {
        "extra_instruction": (
            "Strongly preserve the reference poster's structure. "
            "Explicitly mention visible layout cues such as title block location, product placement, "
            "discount or promo text area, and CTA/button area if present. "
            "Describe the poster as a clean commercial e-commerce advertisement. "
            "Use the target product as the main advertised object."
        )
    },
    "balanced": {
        "extra_instruction": (
            "Balance reference-style transfer and product fidelity. "
            "Keep the reference poster's overall composition, palette, and advertising mood, "
            "while making the target product the clear main subject. "
            "Preserve a clean shopping-poster style with a readable headline area."
        )
    },
    "product_focus": {
        "extra_instruction": (
            "Prioritize target product fidelity and visibility. "
            "Use the reference poster mainly for palette, mood, and broad composition cues. "
            "Keep the poster clean, readable, and product-centric. "
            "If needed, simplify the layout while maintaining an advertisement-poster feeling."
        )
    },
}


def get_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def safe_open_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode in ("P", "RGBA", "LA"):
        img = img.convert("RGBA").convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def get_pair_id(row, idx: int) -> str:
    if "pair_id" in row and pd.notna(row["pair_id"]):
        return str(row["pair_id"]).zfill(4)
    return str(idx + 1).zfill(4)


def choose_canvas_size(ref_w: int, ref_h: int, backend: str = "sdxl", max_side_override: int | None = None) -> Tuple[int, int]:
    if max_side_override is not None:
        max_side = max_side_override
    else:
        # Smaller defaults to reduce VRAM; final deliverable is resized to 224x224 anyway.
        max_side = 768 if backend == "sdxl" else 640
    min_side = 512 if backend == "sdxl" else 448
    scale = max_side / max(ref_w, ref_h)
    w = max(min_side, int(round(ref_w * scale / 64) * 64))
    h = max(min_side, int(round(ref_h * scale / 64) * 64))
    return w, h


def first_n_words(text: str, n: int) -> str:
    words = text.strip().split()
    return " ".join(words[:n]).strip()


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace('"', "").replace("'", "")
    return s


def parse_qwen_lines(decoded: str):
    prompt = None
    negative = None

    for line in decoded.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.upper().startswith("PROMPT:"):
            prompt = line.split(":", 1)[1].strip()
        elif line.upper().startswith("NEGATIVE:"):
            negative = line.split(":", 1)[1].strip()

    # fallback
    if not prompt:
        lines = [x.strip() for x in decoded.splitlines() if x.strip()]
        for line in lines:
            if not line.lower().startswith("line 1") and not line.lower().startswith("line 2"):
                prompt = line
                break

    if not negative:
        negative = "blurry, distorted product, unreadable text, cluttered layout"

    if not prompt:
        raise ValueError(f"Qwen output empty / unparsable:\n{decoded}")

    return {
        "prompt": prompt,
        "negative_prompt": negative,
    }

def truncate_for_pipe(pipe, text: str, max_length: int = 75) -> str:
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None:
        return text
    ids = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).input_ids
    return tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def truncate_for_sdxl(pipe, text: str, max_length: int = 75) -> str:
    tok1 = getattr(pipe, "tokenizer", None)
    tok2 = getattr(pipe, "tokenizer_2", None)
    out = text
    if tok1 is not None:
        ids1 = tok1(text, truncation=True, max_length=max_length, return_tensors="pt").input_ids
        out = tok1.batch_decode(ids1, skip_special_tokens=True)[0]
    if tok2 is not None:
        ids2 = tok2(out, truncation=True, max_length=max_length, return_tensors="pt").input_ids
        out = tok2.batch_decode(ids2, skip_special_tokens=True)[0]
    return out


class QwenPromptGenerator:
    def __init__(self, model_name: str):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        dtype = get_dtype()

        if HAS_QWEN3_NATIVE:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        self.model.eval()

    def build_instruction(self, product_title: str, prompt_style: str) -> str:
        style_inst = PROMPT_STYLES[prompt_style]["extra_instruction"]

        return f"""
You are creating a short diffusion prompt for a product advertisement poster.

Image A: reference poster
Image B: target product image

Target product title:
{product_title}

Your task:
1. Analyze Image A and identify:
   - overall composition
   - title/headline area
   - main product area
   - discount / promo text area if visible
   - CTA / button area if visible
   - color palette and advertising mood
2. Use Image B as the advertised product.
3. Write one short generation prompt for a new poster.

Requirements:
- The poster must advertise the target product from Image B.
- The prompt should explicitly mention 3 to 5 layout cues from Image A.
- The prompt should mention a headline/title area.
- The prompt should mention a promotional shopping-poster or commercial-ad feeling.
- The prompt should emphasize that the target product from Image B should be clearly shown.
- The main prompt must be under 50 English words.
- The negative prompt must be under 18 English words.
- Do not use JSON.
- Do not explain.
- Output exactly 2 lines.

Output format:
PROMPT: ...
NEGATIVE: ...

Instruction strategy:
{style_inst}
""".strip()

    def __call__(self, ref_img: Image.Image, product_img: Image.Image, product_title: str, prompt_style: str) -> Dict[str, str]:
        instruction = self.build_instruction(product_title, prompt_style)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ref_img},
                    {"type": "image", "image": product_img},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        try:
            chat_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[chat_text],
                images=[ref_img, product_img],
                return_tensors="pt",
                padding=True,
            )
        except Exception:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

        for k, v in inputs.items():
            if hasattr(v, "to"):
                inputs[k] = v.to(self.model.device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
            )

        input_len = inputs["input_ids"].shape[1]
        gen_ids = out[:, input_len:]
        decoded = self.processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return parse_qwen_lines(decoded)


class PosterGenerator:
    def __init__(self, args):
        self.args = args
        self.dtype = get_dtype()
        self.prompt_model = QwenPromptGenerator(args.qwen_model)

        base_model = args.sdxl_model if args.backend == "sdxl" else args.sd15_model
        pipe_kwargs = dict(torch_dtype=self.dtype)
        if args.backend == "sdxl":
            pipe_kwargs["use_safetensors"] = True
        else:
            pipe_kwargs["safety_checker"] = None

        self.pipe = AutoPipelineForText2Image.from_pretrained(base_model, **pipe_kwargs)

        self.pipe.load_ip_adapter(
            args.ip_adapter_repo,
            subfolder=args.ip_adapter_subfolder,
            weight_name=args.ip_adapter_weight,
        )
        self.pipe.set_ip_adapter_scale(args.ip_adapter_scale)

        # IMPORTANT:
        # IP-Adapter has known compatibility issues with some memory-saving methods.
        # Keep only the safer VAE slicing by default.
        if args.low_vram:
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
            if hasattr(self.pipe, "enable_vae_tiling") and args.use_vae_tiling:
                self.pipe.enable_vae_tiling()

            if args.allow_risky_memory_opts:
                # Only enable these if you explicitly want to test them.
                if hasattr(self.pipe, "enable_attention_slicing"):
                    self.pipe.enable_attention_slicing()
                if hasattr(self.pipe, "enable_model_cpu_offload"):
                    self.pipe.enable_model_cpu_offload()
                else:
                    self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    def _truncate_prompt(self, text: str) -> str:
        if self.args.backend == "sdxl":
            return truncate_for_sdxl(self.pipe, text)
        return truncate_for_pipe(self.pipe, text)

    def generate_one(self, ref_img: Image.Image, product_img: Image.Image, prompt: str, negative_prompt: str) -> Image.Image:
        width, height = choose_canvas_size(
            ref_img.width,
            ref_img.height,
            self.args.backend,
            self.args.max_side,
        )
        prompt = self._truncate_prompt(prompt)
        negative_prompt = self._truncate_prompt(negative_prompt)

        common_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.args.num_steps,
            guidance_scale=self.args.guidance_scale,
            width=width,
            height=height,
        )

        # Preferred form: single PIL image
        try:
            result = self.pipe(ip_adapter_image=product_img, **common_kwargs)
            return result.images[0]
        except Exception as e1:
            # Secondary fallback: list of one image
            try:
                result = self.pipe(ip_adapter_image=[product_img], **common_kwargs)
                return result.images[0]
            except Exception as e2:
                raise RuntimeError(
                    "IP-Adapter generation failed in both single-image and list-image modes. "
                    f"single mode error={repr(e1)} | list mode error={repr(e2)}"
                )

    def run(self):
        data_root = Path(self.args.data_root)
        ref_dir = data_root / "ref"
        product_dir = data_root / "product"
        pairs_csv = data_root / "pairs.csv"

        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(pairs_csv)
        prompt_records = []

        for idx, row in df.iterrows():
            pair_id = get_pair_id(row, idx)
            ref_path = ref_dir / str(row["ref_image"])
            product_path = product_dir / str(row["product_image"])
            product_title = str(row["product_title"])

            if not ref_path.exists():
                print(f"[skip] missing ref image: {ref_path}")
                continue
            if not product_path.exists():
                print(f"[skip] missing product image: {product_path}")
                continue

            try:
                ref_img = safe_open_rgb(ref_path)
                product_img = safe_open_rgb(product_path)

                qwen_result = self.prompt_model(
                    ref_img=ref_img,
                    product_img=product_img,
                    product_title=product_title,
                    prompt_style=self.args.prompt_style,
                )
                prompt = qwen_result["prompt"]
                negative_prompt = qwen_result["negative_prompt"]

                poster = self.generate_one(ref_img, product_img, prompt, negative_prompt)
                poster_224 = poster.resize((224, 224), Image.LANCZOS)

                save_path = out_dir / f"{pair_id}.jpg"
                poster_224.save(save_path, quality=95)

                prompt_records.append({
                    "pair_id": pair_id,
                    "ref_image": str(row["ref_image"]),
                    "product_image": str(row["product_image"]),
                    "product_title": product_title,
                    "prompt_style": self.args.prompt_style,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                })
                print(f"[ok] {save_path}")

            except Exception as e:
                print(f"[fail] pair_id={pair_id} err={e}")
                traceback.print_exc()

        prompt_json_path = Path(self.args.prompts_json)
        prompt_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prompt_json_path, "w", encoding="utf-8") as f:
            json.dump(prompt_records, f, ensure_ascii=False, indent=2)
        print(f"Saved prompts to {prompt_json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompts_json", type=str, required=True)

    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--backend", type=str, choices=["sdxl", "sd15"], default="sd15")
    parser.add_argument("--sdxl_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--sd15_model", type=str, default="runwayml/stable-diffusion-v1-5")

    parser.add_argument("--ip_adapter_repo", type=str, default="h94/IP-Adapter")
    parser.add_argument("--ip_adapter_subfolder", type=str, default="models")
    parser.add_argument("--ip_adapter_weight", type=str, default="ip-adapter_sd15.bin")
    parser.add_argument("--ip_adapter_scale", type=float, default=0.65)

    parser.add_argument("--prompt_style", type=str, choices=list(PROMPT_STYLES.keys()), default="balanced")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=6.5)

    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--allow_risky_memory_opts", action="store_true",
                        help="Enable attention slicing / cpu offload even though they can break IP-Adapter.")
    parser.add_argument("--use_vae_tiling", action="store_true")
    parser.add_argument("--max_side", type=int, default=None,
                        help="Override internal generation max side. Good low-VRAM values: sd15=640, sdxl=768.")

    args = parser.parse_args()

    if args.backend == "sdxl":
        if args.ip_adapter_subfolder == "models":
            args.ip_adapter_subfolder = "sdxl_models"
        if args.ip_adapter_weight == "ip-adapter_sd15.bin":
            args.ip_adapter_weight = "ip-adapter_sdxl.bin"

    generator = PosterGenerator(args)
    generator.run()


if __name__ == "__main__":
    main()
