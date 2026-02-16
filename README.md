# Dispersion4Q

Official implementation based on the PyTorch and Hugging Face Transformers libraries.

# Installation
All experiments are tested with Python 3.8, torch 2.4.0

### Install Requirements
```
pip install -r requirements.txt
```

### UPD: Install all dependencies with uv:
```
uv venv .venv --python 3.8
source .venv/bin/activate
uv sync
```

### Optional: Prepare a separate vLLM environment
Use an isolated env for modern vLLM dependencies:
```
uv venv .venv_vllm --python /usr/bin/python3.11
UV_CACHE_DIR=$(pwd)/.uv-cache \
UV_PROJECT_ENVIRONMENT=$(pwd)/.venv_vllm \
uv sync --project environments/vllm
source .venv_vllm/bin/activate
```

### Install Codebase
```
cd dispersion4Q
pip install -U pip setuptools==69.5.1
pip install -e .
```

# Datasets
You can find datasets here:

[WMT24_Testset](./src/llama_recipes/customer_data/wmt24_testset)

# Quick Run

## Training & Inference

You can reproduce the results of applying calibration on TowerInstruct-Mistral-7B in Table-1. Training will takes around 1 GPU hour on H100. 
```
cd experiments
export HF_TOKEN=hf_xxx_your_token
bash run.sh
```

`HF_TOKEN` is required for gated model access (e.g. `Unbabel/XCOMET-XXL`) in Slurm jobs.

You will find results, including your hypos for testing data (WMT24++) and quality scores across multiple metrics, in `./results`.

## vLLM Quantization and Calibration

For vLLM runs with quantization configs (for example GPTQ/AWQ), the pipeline does:

1. Merge LoRA adapter into the base model (`experiments/merge_lora_for_vllm.py`).
2. Quantize the merged model (`experiments/quantize_for_vllm.py`) using `llmcompressor` in `.venv_vllm`.
3. Run vLLM inference on the quantized checkpoint.

Quantization config is passed as the 5th argument to `experiments/train_dispersion.sh`.
Examples:

```bash
bash experiments/run_vllm_gptq.sh
bash experiments/run_vllm_awq.sh
```

Calibration details:
- Current calibration data is built from source-side files in `src/llama_recipes/customer_data/<dataset>/test`.
- The script writes a temporary JSONL with schema `{"text": ...}` and uses that for `llmcompressor.oneshot`.
- Default language pairs are `en-ru,en-zh` with up to `512` samples.
- For better quantization fidelity, calibration data should likely be formatted as full translation traces (instruction + source + target-side generation context), not source text only.

TODO:
- Add training dataset calibration support (and make train/test calibration source configurable in `experiments/quantize_for_vllm.py`).
