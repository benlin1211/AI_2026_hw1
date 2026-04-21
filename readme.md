# AI 2026 HW1
## Environment Setup:
- Task 1 
```
conda create -n ipadapter python=3.10 -y
conda activate ipadapter
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-fidelity
pip install -U git+https://github.com/huggingface/transformers
pip install -U git+https://github.com/huggingface/diffusers
pip install -U accelerate peft huggingface_hub safetensors sentencepiece

```
- Task 2
```
conda create -n aimm python=3.10 -y
conda activate aimm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U transformers accelerate
pip install -r requirements_ocr_easyocr.txt
pip install -r requirements.txt

```

## Run Task 1:
- Generation:
```
cd hw1_D14922031_task1
python hw1_D14922031_task1_code.py \
  --data_root ./ \
  --output_dir ./runs_layout_strict/images \
  --prompts_json ./runs_layout_strict/prompts.json \
  --backend sdxl \
  --prompt_style layout_strict \
  --low_vram \
  --max_side 640

python hw1_D14922031_task1_code.py \
  --data_root ./ \
  --output_dir ./runs_balanced/images \
  --prompts_json ./runs_balanced/prompts.json \
  --backend sdxl \
  --prompt_style balanced \
  --low_vram \
  --max_side 640

python hw1_D14922031_task1_code.py \
  --data_root ./ \
  --output_dir ./runs_product_focus/images \
  --prompts_json ./runs_product_focus/prompts.json \
  --backend sdxl \
  --prompt_style product_focus \
  --low_vram \
  --max_side 640
```
- Evaluation:
```
python evaluate_posters.py \
  --data_root ./ \
  --generated_dir ./runs_layout_strict/images \
  --prompts_json ./runs_layout_strict/prompts.json \
  --output_csv ./runs_layout_strict/metrics.csv \
  --output_json ./runs_layout_strict/summary.json

python evaluate_posters.py \
  --data_root ./ \
  --generated_dir ./runs_balanced/images \
  --prompts_json ./runs_balanced/prompts.json \
  --output_csv ./runs_balanced/metrics.csv \
  --output_json ./runs_balanced/summary.json

python evaluate_posters.py \
  --data_root ./ \
  --generated_dir ./runs_product_focus/images \
  --prompts_json ./runs_product_focus/prompts.json \
  --output_csv ./runs_product_focus/metrics.csv \
  --output_json ./runs_product_focus/summary.json
```

## Run Task 2:
- OCR cache:
```
cd hw1_D14922031_task2
python ocr_structured_v3_roles.py \
  --pdf_path AI.pdf \
  --dpi 200 \
  --languages en \
  --use_gpu \
  --annotate_all_pages \
  --annotation_model_name Qwen/Qwen2.5-VL-3B-Instruct \
  --annotation_max_new_tokens 256
```
- RAG execution:
```
python rag_page_locator_vlm_verifier_widek.py \
  --pdf_path AI.pdf \
  --question_path HW1_questions.json \
  --output_path submission.csv \
  --debug_path debug_vlm_widek.csv \
  --trace_path trace_vlm_widek.jsonl \
  --top_k_candidates 24 \
  --top_k_vlm 8 \
  --load_in_4bit
```