#!/usr/bin/env bash

# Gemma-3-4B-IT raw-model evaluation (no training), vLLM beam=1.
# Usage:
#   bash experiments/gemma-3-12b-it/run_vllm_eval_raw_model.sh [tower1|rare|custom]
# Optional env:
#   LANG_DIRECTIONS_CUSTOM=en-is,en-et  (when LANG_SET=custom)
#   VLLM_ENV_PATH=.venv_vllm
#   MODEL_NAME=google/gemma-3-4b-it
#   BEAM_SIZE=1
#   NUM_GPUS=2
#   VLLM_TENSOR_PARALLEL_SIZE=2
#   MAX_NEW_TOKENS=1024  (optional override; default comes from model profile)
#   VLLM_MAX_MODEL_LEN=2048
#   VLLM_MAX_NUM_BATCHED_TOKENS=512

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
    EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    cd "${EXPERIMENTS_DIR}"

    LANG_SET_ARG="${1:-${LANG_SET:-tower1}}"
    NUM_GPUS="${NUM_GPUS:-1}"
    VLLM_TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-${NUM_GPUS}}"
    echo "Submitting Gemma raw-model vLLM eval with LANG_SET=${LANG_SET_ARG}"
    sbatch \
      --export=ALL,LANG_SET="${LANG_SET_ARG}",NUM_GPUS="${NUM_GPUS}",VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TP_SIZE}" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=8 \
      --mem=100G \
      --partition=gpu_h100 \
      --gres=gpu:"${NUM_GPUS}" \
      --time=01-00:00:00 \
      --job-name=gemma3-raw-vllm \
      "${SCRIPT_PATH}"
    exit 0
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_DIR="${SUBMIT_DIR}"
[[ -f "${PROJECT_DIR}/pyproject.toml" ]] || PROJECT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

source "${PROJECT_DIR}/.venv/bin/activate"

unset HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME TORCH_HOME
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

LANG_SET="${LANG_SET:-tower1}"
LANG_DIRECTIONS_TOWER1=("en-de" "en-es" "en-ru" "en-zh" "en-fr" "en-nl" "en-it" "en-pt" "en-ko")
LANG_DIRECTIONS_RARE=("en-ru" "en-is" "en-et" "en-lv" "en-sl")

case "${LANG_SET}" in
    tower1)
        LANG_DIRECTIONS=("${LANG_DIRECTIONS_TOWER1[@]}")
        ;;
    rare)
        LANG_DIRECTIONS=("${LANG_DIRECTIONS_RARE[@]}")
        ;;
    custom)
        if [[ -z "${LANG_DIRECTIONS_CUSTOM:-}" ]]; then
            echo "ERROR: LANG_SET=custom requires LANG_DIRECTIONS_CUSTOM as comma-separated pairs." >&2
            echo "Example: LANG_DIRECTIONS_CUSTOM=en-is,en-et,en-lv,en-sl" >&2
            exit 1
        fi
        IFS=',' read -r -a LANG_DIRECTIONS <<< "${LANG_DIRECTIONS_CUSTOM}"
        ;;
    *)
        echo "ERROR: LANG_SET must be one of: tower1, rare, custom. Got '${LANG_SET}'." >&2
        exit 1
        ;;
esac

LANG_PAIRS_CSV="$(IFS=,; echo "${LANG_DIRECTIONS[*]}")"

TEST_DATASET_DEFAULT="${TEST_DATASET_DEFAULT:-wmt24_testset}"
TEST_DATASET_RARE="${TEST_DATASET_RARE:-wmt24pp_testset}"
if [[ "${LANG_SET}" == "rare" ]]; then
    TEST_DATASET="${TEST_DATASET_RARE}"
else
    TEST_DATASET="${TEST_DATASET_DEFAULT}"
fi

MODEL_NAME="${MODEL_NAME:-google/gemma-3-4b-it}"
MODEL_TAG="${MODEL_NAME//\//-}"
BEAM_SIZE="${BEAM_SIZE:-1}"
VLLM_ENV_PATH="${VLLM_ENV_PATH:-.venv_vllm}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
VLLM_TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-${NUM_GPUS:-1}}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
RESULT_TAG="raw-vllm-beam${BEAM_SIZE}"
BASE_SYS="results/${MODEL_TAG}/${TEST_DATASET}/${RESULT_TAG}/${LANG_SET}"

mkdir -p "${BASE_SYS}"

echo "Model: ${MODEL_NAME}"
echo "LANG_SET: ${LANG_SET}"
echo "Language pairs: ${LANG_PAIRS_CSV}"
echo "Dataset: ${TEST_DATASET}"
echo "Output dir: ${BASE_SYS}"
if [[ -n "${MAX_NEW_TOKENS}" ]]; then
    echo "Override max_new_tokens: ${MAX_NEW_TOKENS}"
else
    echo "max_new_tokens: from model profile/default"
fi
echo "Tensor parallel size: ${VLLM_TP_SIZE}"
if [[ -n "${VLLM_MAX_MODEL_LEN}" ]]; then
    echo "Override vLLM max_model_len: ${VLLM_MAX_MODEL_LEN}"
fi
if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
    echo "Override vLLM max_num_batched_tokens: ${VLLM_MAX_NUM_BATCHED_TOKENS}"
fi

INFER_EXTRA_ARGS=()
if [[ -n "${VLLM_MAX_MODEL_LEN}" ]]; then
    INFER_EXTRA_ARGS+=(--vllm_max_model_len "${VLLM_MAX_MODEL_LEN}")
fi
if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
    INFER_EXTRA_ARGS+=(--vllm_max_num_batched_tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}")
fi
if [[ -n "${MAX_NEW_TOKENS}" ]]; then
    INFER_EXTRA_ARGS+=(--max_new_tokens "${MAX_NEW_TOKENS}")
fi

python experiments/inference_formal.py \
    --model_name "${MODEL_NAME}" \
    --dataset "${TEST_DATASET}" \
    --do_sample False \
    --output_dir "${BASE_SYS}" \
    --lang_pairs "${LANG_PAIRS_CSV}" \
    --beam_size "${BEAM_SIZE}" \
    --inference_backend vllm \
    --vllm_env_path "${VLLM_ENV_PATH}" \
    --vllm_tensor_parallel_size "${VLLM_TP_SIZE}" \
    "${INFER_EXTRA_ARGS[@]}"

BASE_SRC="${PROJECT_DIR}/src/llama_recipes/customer_data/${TEST_DATASET}/test"
for LANG_DIR in "${LANG_DIRECTIONS[@]}"; do
    SRC_LANG="${LANG_DIR%%-*}"
    TGT_LANG="${LANG_DIR##*-}"

    SRC_FILE="${BASE_SRC}/${LANG_DIR}/test.${LANG_DIR}.${SRC_LANG}"
    TGT_FILE="${BASE_SRC}/${LANG_DIR}/test.${LANG_DIR}.${TGT_LANG}"
    SYS_FILE="${BASE_SYS}/${LANG_DIR}/hyp.${LANG_DIR}.${TGT_LANG}"

    if [[ ! -f "${SYS_FILE}" ]]; then
        echo "[WARN] Missing system output for ${LANG_DIR}: ${SYS_FILE}. Skipping scoring." >&2
        continue
    fi

    COMET_SCORE_FILE="${BASE_SYS}/${LANG_DIR}/comet.score"
    XCOMET_SCORE_FILE="${BASE_SYS}/${LANG_DIR}/xcomet.score"
    KIWI_SCORE_FILE="${BASE_SYS}/${LANG_DIR}/kiwi.score"
    KIWI_XL_SCORE_FILE="${BASE_SYS}/${LANG_DIR}/kiwi-xl.score"
    KIWI_XXL_SCORE_FILE="${BASE_SYS}/${LANG_DIR}/kiwi-xxl.score"
    mkdir -p "$(dirname "${COMET_SCORE_FILE}")"

    echo "Scoring ${LANG_DIR}..."
    comet-score -s "${SRC_FILE}" -t "${SYS_FILE}" -r "${TGT_FILE}" --model Unbabel/wmt22-comet-da >> "${COMET_SCORE_FILE}"

    if [[ -n "${HF_TOKEN:-}" ]]; then
        if ! comet-score -s "${SRC_FILE}" -t "${SYS_FILE}" --model Unbabel/XCOMET-XXL >> "${XCOMET_SCORE_FILE}"; then
            echo "[WARN] XCOMET-XXL failed for ${LANG_DIR}. Continuing." >&2
        fi
    else
        echo "[WARN] HF_TOKEN is not set; skipping XCOMET-XXL for ${LANG_DIR}." >&2
    fi

    comet-score -s "${SRC_FILE}" -t "${SYS_FILE}" --model Unbabel/wmt22-cometkiwi-da >> "${KIWI_SCORE_FILE}"
    comet-score -s "${SRC_FILE}" -t "${SYS_FILE}" --model Unbabel/wmt23-cometkiwi-da-xl >> "${KIWI_XL_SCORE_FILE}"
    comet-score -s "${SRC_FILE}" -t "${SYS_FILE}" --model Unbabel/wmt23-cometkiwi-da-xxl >> "${KIWI_XXL_SCORE_FILE}"
done

echo "Raw-model vLLM evaluation completed."
