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

export HF_HUB_CACHE="${HF_HUB_CACHE:-${PROJECT_DIR}/.cache/huggingface}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
mkdir -p "${HF_HUB_CACHE}"

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
        comet-score -s $SRC_FILE -t $SYS_FILE --model Unbabel/XCOMET-XXL >> $XCOMET_SCORE_FILE
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

echo "LR is set to: $LR"
echo "Base_model is set to: $BASE_MODEL"

SETTING=${LR}-test
TEST_DATASET=wmt24_testset
CKP_DIR="${PROJECT_DIR}/experiments/checkpoints"

echo "CKP: $CKP_DIR/$BASE_MODEL/dispersion4Q/${SETTING}"
echo "RESULTS: results/$BASE_MODEL/dispersion4Q/${TEST_DATASET}/${SETTING}-beam5"
echo "SCORES: scores/$BASE_MODEL/dispersion4Q/${SETTING}/0/wmt-qe-22-test"

# Train
python -m llama_recipes.finetuning --use_peft --peft_method lora \
        --model_name Unbabel/$BASE_MODEL \
        --output_dir $CKP_DIR/$BASE_MODEL/${SETTING} \
        --dataset flores_dataset \
        --batching_strategy padding \
        --num_epochs 1 \
        --lr $LR \
        --batch_size_training 32 \
        --val_batch_size 32 \
        --gradient_accumulation_steps 8 \
        --lang_pairs "en-ru,en-zh" \
        --use_wandb

# Test
for EPOCH in 0; do
    BASE_SYS=results/$BASE_MODEL/${TEST_DATASET}/${SETTING}-beam5/${EPOCH}
    python experiments/inference_formal.py --model_name Unbabel/$BASE_MODEL \
            --peft_model $CKP_DIR/$BASE_MODEL/${SETTING}/${EPOCH} \
            --dataset ${TEST_DATASET} \
            --val_batch_size 8 \
            --do_sample False \
            --output_dir ${BASE_SYS} \
            --lang_pairs en-ru,en-zh \
            --beam_size 5
    evaluate_lang_directions ${TEST_DATASET} ${BASE_SYS}
done
