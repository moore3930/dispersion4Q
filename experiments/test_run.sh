#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=01-00:00:00

#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_DIR="${SUBMIT_DIR}"
[[ -f "${PROJECT_DIR}/pyproject.toml" ]] || PROJECT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}/experiments"
source "${PROJECT_DIR}/.venv/bin/activate"

export HF_HUB_CACHE="${HF_HUB_CACHE:-${PROJECT_DIR}/.cache/huggingface}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
mkdir -p "${HF_HUB_CACHE}"

python test_pipeline.py
