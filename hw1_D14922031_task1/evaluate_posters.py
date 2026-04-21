import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from torchmetrics.image.kid import KernelInceptionDistance


def _extract_tensor_feature(self, outputs, preferred_attr=None):
    if torch.is_tensor(outputs):
        return outputs

    if preferred_attr is not None and hasattr(outputs, preferred_attr):
        val = getattr(outputs, preferred_attr)
        if torch.is_tensor(val):
            return val

    for attr in ["image_embeds", "text_embeds", "pooler_output", "last_hidden_state"]:
        if hasattr(outputs, attr):
            val = getattr(outputs, attr)
            if torch.is_tensor(val):
                if attr == "last_hidden_state":
                    return val[:, 0]
                return val

    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if torch.is_tensor(item):
                return item
            if hasattr(item, "pooler_output") and torch.is_tensor(item.pooler_output):
                return item.pooler_output
            if hasattr(item, "last_hidden_state") and torch.is_tensor(item.last_hidden_state):
                return item.last_hidden_state[:, 0]

    raise TypeError(f"Cannot extract tensor feature from output type: {type(outputs)}")


def get_pair_id(row, idx):
    if "pair_id" in row and pd.notna(row["pair_id"]):
        return str(row["pair_id"]).zfill(4)
    return str(idx + 1).zfill(4)


def load_prompt_map(prompts_json_path):
    with open(prompts_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mp = {}
    for item in data:
        mp[str(item["pair_id"]).zfill(4)] = item["prompt"]
    return mp


def image_to_uint8_tensor(img: Image.Image, size=299):
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # [0,1]
    ])
    x = tfm(img) * 255.0
    x = x.clamp(0, 255).to(torch.uint8)
    return x


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
        self.clip_model = CLIPModel.from_pretrained(args.clip_model).to(self.device)
        self.clip_model.eval()

        self.kid = KernelInceptionDistance(
            subset_size=args.kid_subset_size,
            normalize=False,
        ).to(self.device)

    def encode_image(self, img: Image.Image):
        inputs = self.clip_processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
            feat = vision_outputs.pooler_output
            if hasattr(self.clip_model, "visual_projection") and self.clip_model.visual_projection is not None:
                feat = self.clip_model.visual_projection(feat)

        feat = F.normalize(feat, dim=-1)
        return feat


    def encode_text(self, text: str):
        inputs = self.clip_processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            text_outputs = self.clip_model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            feat = text_outputs.pooler_output
            if hasattr(self.clip_model, "text_projection") and self.clip_model.text_projection is not None:
                feat = self.clip_model.text_projection(feat)

        feat = F.normalize(feat, dim=-1)
        return feat

    def run(self):
        data_root = Path(self.args.data_root)
        gen_dir = Path(self.args.generated_dir)
        ref_dir = data_root / "ref"
        product_dir = data_root / "product"
        pairs_csv = data_root / "pairs.csv"

        df = pd.read_csv(pairs_csv)
        prompt_map = load_prompt_map(self.args.prompts_json)

        rows = []
        image_image_scores = []
        image_text_scores = []

        for idx, row in df.iterrows():
            pair_id = get_pair_id(row, idx)
            gen_path = gen_dir / f"{pair_id}.jpg"
            ref_path = ref_dir / str(row["ref_image"])
            product_path = product_dir / str(row["product_image"])

            if not gen_path.exists():
                print(f"[skip] missing generated image: {gen_path}")
                continue
            if not ref_path.exists():
                print(f"[skip] missing ref image: {ref_path}")
                continue
            if not product_path.exists():
                print(f"[skip] missing product image: {product_path}")
                continue

            prompt = prompt_map.get(pair_id, str(row["product_title"]))

            gen_img = Image.open(gen_path).convert("RGB")
            ref_img = Image.open(ref_path).convert("RGB")
            product_img = Image.open(product_path).convert("RGB")

            # CLIP image-image similarity
            gen_feat = self.encode_image(gen_img)
            prod_feat = self.encode_image(product_img)
            sim_img = torch.sum(gen_feat * prod_feat, dim=-1).item()

            # CLIP image-text similarity
            text_feat = self.encode_text(prompt)
            sim_txt = torch.sum(gen_feat * text_feat, dim=-1).item()

            image_image_scores.append(sim_img)
            image_text_scores.append(sim_txt)

            # KID
            gen_kid = image_to_uint8_tensor(gen_img).unsqueeze(0).to(self.device)
            ref_kid = image_to_uint8_tensor(ref_img).unsqueeze(0).to(self.device)

            self.kid.update(ref_kid, real=True)
            self.kid.update(gen_kid, real=False)

            rows.append({
                "pair_id": pair_id,
                "generated_path": str(gen_path),
                "ref_image": str(row["ref_image"]),
                "product_image": str(row["product_image"]),
                "clip_visual_similarity": sim_img,
                "clip_prompt_similarity": sim_txt,
                "prompt": prompt,
            })

            print(f"[ok] evaluated {pair_id}")

        kid_mean, kid_std = self.kid.compute()

        summary = {
            "num_samples": len(rows),
            "clip_visual_similarity_mean": float(sum(image_image_scores) / max(1, len(image_image_scores))),
            "clip_prompt_similarity_mean": float(sum(image_text_scores) / max(1, len(image_text_scores))),
            "kid_mean": float(kid_mean.item()),
            "kid_std": float(kid_std.item()),
        }

        out_csv = Path(self.args.output_csv)
        out_json = Path(self.args.output_json)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_json.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(rows).to_csv(out_csv, index=False)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n=== Summary ===")
        print(json.dumps(summary, indent=2))
        print(f"\nSaved per-sample metrics to {out_csv}")
        print(f"Saved summary metrics to {out_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--generated_dir", type=str, required=True)
    parser.add_argument("--prompts_json", type=str, required=True)

    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)

    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--kid_subset_size", type=int, default=50)

    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()