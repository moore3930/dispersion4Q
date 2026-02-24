#!/usr/bin/env bash
# Gemma-3-4B-IT with KIWI-XXL (vLLM inference, beam=1)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${EXPERIMENTS_DIR}"

LANG_SET_ARG="${1:-${LANG_SET:-tower1}}"
echo "Submitting Gemma run_vllm with LANG_SET=${LANG_SET_ARG}"
sbatch --export=ALL,LANG_SET="${LANG_SET_ARG}" train_dispersion.sh 5e-5 google/gemma-3-4b-it 1 vllm
