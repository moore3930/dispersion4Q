#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=01-00:00:00

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_DIR="${SUBMIT_DIR}"
[[ -f "${PROJECT_DIR}/pyproject.toml" ]] || PROJECT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"
MAIN_PYTHON="${PROJECT_DIR}/.venv/bin/python"

unset HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME TORCH_HOME
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is required for gated model access (Unbabel/XCOMET-XXL)." >&2
    echo "Run with: sbatch --export=ALL,HF_TOKEN=hf_xxx experiments/train_dispersion.sh <LR> <BASE_MODEL> [BEAM_SIZE] [INFERENCE_BACKEND] [QUANTIZATION_CONFIG]" >&2
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

evaluate_lang_directions() {
    # Parameters
    local TEST_DATASET="$1"  # Test dataset name
    local BASE_SYS="$2"     # Base system directory

    # Define language directions (can customize or pass as parameter if needed)
    # local LANG_DIRECTIONS=("en-de" "en-es" "en-ru" "en-zh" "en-fr" "en-nl" "en-it" "en-pt" "en-ko") # tower-1 langs
    local LANG_DIRECTIONS=("en-ru" "en-zh") # testing

    # Define base source and target directories
    local BASE_SRC="${PROJECT_DIR}/src/llama_recipes/customer_data/${TEST_DATASET}/test"
    local BASE_TGT=$BASE_SRC
    # Loop through each language direction
    for LANG_DIR in "${LANG_DIRECTIONS[@]}"; do
        # Extract source and target language codes
        local SRC_LANG=$(echo $LANG_DIR | cut -d'-' -f1)
        local TGT_LANG=$(echo $LANG_DIR | cut -d'-' -f2)

        # Define the file paths
        local SRC_FILE="${BASE_SRC}/${LANG_DIR}/test.${LANG_DIR}.${SRC_LANG}"
        local TGT_FILE="${BASE_TGT}/${LANG_DIR}/test.${LANG_DIR}.${TGT_LANG}"
        local SYS_FILE="${BASE_SYS}/${LANG_DIR}/hyp.${LANG_DIR}.${TGT_LANG}"

        # Define the output score files
        local COMET_SCORE_FILE="./${BASE_SYS}/${LANG_DIR}/comet.score"
        local XCOMET_SCORE_FILE="./${BASE_SYS}/${LANG_DIR}/xcomet.score"
        local KIWI_SCORE_FILE="./${BASE_SYS}/${LANG_DIR}/kiwi.score"
        local KIWI_XL_SCORE_FILE="./${BASE_SYS}/${LANG_DIR}/kiwi-xl.score"
        local KIWI_XXL_SCORE_FILE="./${BASE_SYS}/${LANG_DIR}/kiwi-xxl.score"
        mkdir -p "$(dirname "${COMET_SCORE_FILE}")"

        echo "Calculating COMET scores for ${LANG_DIR}..."

        # Run COMET scoring
        comet-score -s $SRC_FILE -t $SYS_FILE -r $TGT_FILE --model Unbabel/wmt22-comet-da >> $COMET_SCORE_FILE
        if ! comet-score -s $SRC_FILE -t $SYS_FILE --model Unbabel/XCOMET-XXL >> $XCOMET_SCORE_FILE; then
            echo "[WARN] XCOMET-XXL failed for ${LANG_DIR} (likely gated-access/token permissions). Continuing." >&2
        fi
        comet-score -s $SRC_FILE -t $SYS_FILE --model Unbabel/wmt22-cometkiwi-da >> $KIWI_SCORE_FILE
        comet-score -s $SRC_FILE -t $SYS_FILE --model Unbabel/wmt23-cometkiwi-da-xl >> $KIWI_XL_SCORE_FILE
        comet-score -s $SRC_FILE -t $SYS_FILE --model Unbabel/wmt23-cometkiwi-da-xxl >> $KIWI_XXL_SCORE_FILE

        echo "Finished ${LANG_DIR}"
    done

    echo "All language directions processed!"
}



################## MAIN ##################

LR=$1
BASE_MODEL=$2
BEAM_SIZE=${3:-5}
INFERENCE_BACKEND=${4:-hf}
QUANTIZATION_CONFIG=${5:-}
SKIP_EXISTING_TRAIN=${SKIP_EXISTING_TRAIN:-1}
VLLM_ENV_PATH=${VLLM_ENV_PATH:-.venv_vllm}
VLLM_PYTHON="${PROJECT_DIR}/${VLLM_ENV_PATH}/bin/python"

if [[ "${INFERENCE_BACKEND}" != "hf" && "${INFERENCE_BACKEND}" != "vllm" ]]; then
    echo "ERROR: INFERENCE_BACKEND must be 'hf' or 'vllm', got '${INFERENCE_BACKEND}'." >&2
    exit 1
fi

if [[ "${INFERENCE_BACKEND}" == "vllm" && "${BEAM_SIZE}" -ne 1 ]]; then
    echo "ERROR: vLLM mode requires BEAM_SIZE=1, got ${BEAM_SIZE}." >&2
    exit 1
fi

if [[ "${INFERENCE_BACKEND}" == "vllm" && ! -x "${VLLM_PYTHON}" ]]; then
    echo "ERROR: vLLM python not found/executable: ${VLLM_PYTHON}" >&2
    exit 1
fi

if [[ -n "${QUANTIZATION_CONFIG}" && "${INFERENCE_BACKEND}" != "vllm" ]]; then
    echo "ERROR: QUANTIZATION_CONFIG is set but INFERENCE_BACKEND is '${INFERENCE_BACKEND}'. Use 'vllm' backend." >&2
    exit 1
fi

if [[ -n "${QUANTIZATION_CONFIG}" && ! -f "${PROJECT_DIR}/${QUANTIZATION_CONFIG}" ]]; then
    echo "ERROR: QUANTIZATION_CONFIG file does not exist: ${PROJECT_DIR}/${QUANTIZATION_CONFIG}" >&2
    exit 1
fi

echo "LR is set to: $LR"
echo "Base_model is set to: $BASE_MODEL"
echo "Beam size is set to: $BEAM_SIZE"
echo "Inference backend is set to: $INFERENCE_BACKEND"
echo "Skip existing training is set to: $SKIP_EXISTING_TRAIN"
if [[ "${INFERENCE_BACKEND}" == "vllm" ]]; then
    echo "vLLM env path is set to: $VLLM_ENV_PATH"
    if [[ -n "${QUANTIZATION_CONFIG}" ]]; then
        echo "Quantization config is set to: $QUANTIZATION_CONFIG"
    fi
fi

SETTING=${LR}-test
TEST_DATASET=wmt24_testset
CKP_DIR="${PROJECT_DIR}/experiments/checkpoints"
RESULT_TAG="${SETTING}-${INFERENCE_BACKEND}-beam${BEAM_SIZE}"
if [[ -n "${QUANTIZATION_CONFIG}" ]]; then
    CONFIG_NAME="$(basename "${QUANTIZATION_CONFIG}")"
    CONFIG_NAME="${CONFIG_NAME%.*}"
    RESULT_TAG="${RESULT_TAG}-${CONFIG_NAME}"
fi
TRAIN_OUTPUT_DIR="$CKP_DIR/$BASE_MODEL/${SETTING}"
EPOCH0_CKPT="${TRAIN_OUTPUT_DIR}/0/adapter_model.safetensors"
CALIBRATION_FILE="${PROJECT_DIR}/data/calibration_data/${TEST_DATASET}.json"

echo "CKP: $TRAIN_OUTPUT_DIR"
echo "RESULTS: results/$BASE_MODEL/dispersion4Q/${TEST_DATASET}/${RESULT_TAG}"
echo "SCORES: scores/$BASE_MODEL/dispersion4Q/${SETTING}/0/wmt-qe-22-test"

# Train
if [[ "${SKIP_EXISTING_TRAIN}" == "1" && -f "${EPOCH0_CKPT}" ]]; then
    echo "Skipping training: found existing checkpoint ${EPOCH0_CKPT}"
else
    ${MAIN_PYTHON} -m llama_recipes.finetuning --use_peft --peft_method lora \
            --model_name Unbabel/$BASE_MODEL \
            --output_dir ${TRAIN_OUTPUT_DIR} \
            --dataset flores_dataset \
            --batching_strategy padding \
            --num_epochs 1 \
            --lr $LR \
            --batch_size_training 32 \
            --val_batch_size 32 \
            --gradient_accumulation_steps 8 \
            --lang_pairs "en-ru,en-zh" \
            --use_wandb
fi

# Test
for EPOCH in 0; do
    BASE_SYS=results/$BASE_MODEL/${TEST_DATASET}/${RESULT_TAG}/${EPOCH}
    if [[ "${INFERENCE_BACKEND}" == "vllm" ]]; then
        ADAPTER_DIR="${TRAIN_OUTPUT_DIR}/${EPOCH}"
        MERGED_MODEL_DIR="${TRAIN_OUTPUT_DIR}/${EPOCH}_merged_vllm"
        ${MAIN_PYTHON} experiments/merge_lora_for_vllm.py \
                --base_model Unbabel/$BASE_MODEL \
                --adapter_dir ${ADAPTER_DIR} \
                --output_dir ${MERGED_MODEL_DIR}

        MODEL_FOR_VLLM="${MERGED_MODEL_DIR}"
        if [[ -n "${QUANTIZATION_CONFIG}" ]]; then
            QUANTIZED_MODEL_DIR="${TRAIN_OUTPUT_DIR}/${EPOCH}_quantized_${CONFIG_NAME}"
            ${VLLM_PYTHON} experiments/quantize_for_vllm.py \
                    --model_dir ${MERGED_MODEL_DIR} \
                    --quantization_config ${QUANTIZATION_CONFIG} \
                    --output_dir ${QUANTIZED_MODEL_DIR} \
                    --calibration_data_path ${CALIBRATION_FILE} \
                    --dataset_name ${TEST_DATASET} \
                    --lang_pairs en-ru,en-zh
            MODEL_FOR_VLLM="${QUANTIZED_MODEL_DIR}"
        fi

        python experiments/inference_formal.py --model_name ${MODEL_FOR_VLLM} \
                --dataset ${TEST_DATASET} \
                --val_batch_size 8 \
                --do_sample False \
                --output_dir ${BASE_SYS} \
                --lang_pairs en-ru,en-zh \
                --beam_size ${BEAM_SIZE} \
                --inference_backend ${INFERENCE_BACKEND} \
                --vllm_env_path ${VLLM_ENV_PATH}
    else
        python experiments/inference_formal.py --model_name Unbabel/$BASE_MODEL \
                --peft_model ${TRAIN_OUTPUT_DIR}/${EPOCH} \
                --dataset ${TEST_DATASET} \
                --val_batch_size 8 \
                --do_sample False \
                --output_dir ${BASE_SYS} \
                --lang_pairs en-ru,en-zh \
                --beam_size ${BEAM_SIZE} \
                --inference_backend ${INFERENCE_BACKEND} \
                --vllm_env_path ${VLLM_ENV_PATH}
    fi
    evaluate_lang_directions ${TEST_DATASET} ${BASE_SYS}
done
