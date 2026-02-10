# Dispersion4Q

Official implementation based on the PyTorch and Hugging Face Transformers libraries.

# Installation
All experiments are tested with Python 3.8, torch 2.4.0

### Install Requirements
```
pip install -r requirements.txt
```

### Install Codebase
```
cd dispersion4Q
pip install -U pip setuptools
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
sh run.sh
```

You will find results, inculding your hypos for testing data (WMT24++) and quality scores across multiple metrics, in ./experiments/results