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
