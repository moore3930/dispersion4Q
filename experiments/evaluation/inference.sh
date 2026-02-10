#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=01-00:00:00

#SBATCH -o /home/dwu18/projects/llama-recipes/experiments/logs/out.infer.%j.o
#SBATCH -e /home/dwu18/projects/llama-recipes/experiments/logs/out.infer.%j.e

source activate py38cuda11

export HF_HUB_CACHE=/gpfs/work4/0/gus20642/dwu18/cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


evaluate_lang_direction() {
    # Parameters
    local TEST_DATASET="$1"   # Test dataset name
    local BASE_SYS="$2"       # Base system directory
    local LANG_PAIR="$3"      # Language pair, e.g., en-zh

    # Extract source and target language codes
    local SRC_LANG=$(echo $LANG_PAIR | cut -d'-' -f1)
    local TGT_LANG=$(echo $LANG_PAIR | cut -d'-' -f2)

    # Define base source and target directories
    local BASE_SRC="/gpfs/work4/0/gus20642/dwu18/project/value_finetuning/src/llama_recipes/customer_data/${TEST_DATASET}/test"
    local BASE_TGT="/gpfs/work4/0/gus20642/dwu18/project/value_finetuning/src/llama_recipes/customer_data/${TEST_DATASET}/test"

    # Define the file paths
    local SRC_FILE="${BASE_SRC}/${LANG_PAIR}/test.${LANG_PAIR}.${SRC_LANG}"
    local TGT_FILE="${BASE_TGT}/${LANG_PAIR}/test.${LANG_PAIR}.${TGT_LANG}"
    local SYS_FILE="${BASE_SYS}/${LANG_PAIR}/hyp.${LANG_PAIR}.${TGT_LANG}"

    # Define the output score files
    local COMET_SCORE_FILE="./${BASE_SYS}/${LANG_PAIR}/comet.score"
    local XCOMET_SCORE_FILE="./${BASE_SYS}/${LANG_PAIR}/xcomet.score"
    local KIWI_SCORE_FILE="./${BASE_SYS}/${LANG_PAIR}/kiwi.score"
    local KIWI_XL_SCORE_FILE="./${BASE_SYS}/${LANG_PAIR}/kiwi-xl.score"
    local KIWI_XXL_SCORE_FILE="./${BASE_SYS}/${LANG_PAIR}/kiwi-xxl.score"

    echo "Calculating COMET scores for ${LANG_PAIR}..."

    # Run COMET scoring
    comet-score -s "$SRC_FILE" -t "$SYS_FILE" -r "$TGT_FILE" --model Unbabel/wmt22-comet-da >> "$COMET_SCORE_FILE"
    comet-score -s "$SRC_FILE" -t "$SYS_FILE" --model Unbabel/XCOMET-XXL >> "$XCOMET_SCORE_FILE"
    comet-score -s "$SRC_FILE" -t "$SYS_FILE" --model Unbabel/wmt22-cometkiwi-da >> "$KIWI_SCORE_FILE"
    comet-score -s "$SRC_FILE" -t "$SYS_FILE" --model Unbabel/wmt23-cometkiwi-da-xl >> "$KIWI_XL_SCORE_FILE"
    comet-score -s "$SRC_FILE" -t "$SYS_FILE" --model Unbabel/wmt23-cometkiwi-da-xxl >> "$KIWI_XXL_SCORE_FILE"

    echo "Finished ${LANG_PAIR}"
}


LANG_PAIR=$1

# Evaluate 7B Model
TEST_DATASET=wmt22_testset
TEST_DATASET=wmt24_testset

:<<!
SETTING=ALMA-v1-7B
BASE_SYS=results/${TEST_DATASET}/${SETTING}-beam5
python ../inference_formal.py --model_name haoranxu/ALMA-7B-Pretrain \
        --peft_model haoranxu/ALMA-7B-Pretrain-LoRA \
        --dataset ${TEST_DATASET} \
        --val_batch_size 8 \
        --do_sample False \
        --output_dir ${BASE_SYS} \
        --lang_pairs ${LANG_PAIR} \
        --beam_size 5

evaluate_lang_direction "$TEST_DATASET" "$BASE_SYS" "$LANG_PAIR"


SETTING=ALMA-R-7B
BASE_SYS=results/${TEST_DATASET}/${SETTING}-beam5
python ../inference_formal.py --model_name haoranxu/ALMA-7B-R \
        --dataset ${TEST_DATASET} \
        --val_batch_size 8 \
        --do_sample False \
        --output_dir ${BASE_SYS} \
        --lang_pairs ${LANG_PAIR} \
        --beam_size 5

evaluate_lang_direction "$TEST_DATASET" "$BASE_SYS" "$LANG_PAIR"
!

SETTING=TowerInstruct-Mistral-7B-v0.2
BASE_SYS=results/${TEST_DATASET}/${SETTING}-beam5
echo $LANG_PAIR $SETTING $BASE_SYS
python ../inference_formal.py --model_name Unbabel/TowerInstruct-Mistral-7B-v0.2 \
        --dataset ${TEST_DATASET} \
        --val_batch_size 8 \
        --do_sample False \
        --output_dir ${BASE_SYS} \
        --lang_pairs ${LANG_PAIR} \
        --beam_size 5

evaluate_lang_direction "$TEST_DATASET" "$BASE_SYS" "$LANG_PAIR"

:<<!
SETTING=TowerInstruct-7B-v0.1
BASE_SYS=results/${TEST_DATASET}/${SETTING}-beam5
echo $LANG_PAIR $SETTING $BASE_SYS
python ../inference_formal.py --model_name Unbabel/TowerInstruct-7B-v0.1 \
        --dataset ${TEST_DATASET} \
        --val_batch_size 8 \
        --do_sample False \
        --output_dir ${BASE_SYS} \
        --lang_pairs ${LANG_PAIR} \
        --beam_size 5

evaluate_lang_direction "$TEST_DATASET" "$BASE_SYS" "$LANG_PAIR"

SETTING=TowerInstruct-7B-v0.2
BASE_SYS=results/${TEST_DATASET}/${SETTING}-beam5
echo $LANG_PAIR $SETTING $BASE_SYS
python ../inference_formal.py --model_name Unbabel/TowerInstruct-7B-v0.2 \
        --dataset ${TEST_DATASET} \
        --val_batch_size 8 \
        --do_sample False \
        --output_dir ${BASE_SYS} \
        --lang_pairs ${LANG_PAIR} \
        --beam_size 5

evaluate_lang_direction "$TEST_DATASET" "$BASE_SYS" "$LANG_PAIR"

SETTING=TowerInstruct-13B-v0.1
BASE_SYS=results/${TEST_DATASET}/${SETTING}-beam5
python ../inference_formal.py --model_name Unbabel/TowerInstruct-13B-v0.1 \
        --dataset ${TEST_DATASET} \
        --val_batch_size 4 \
        --do_sample False \
        --output_dir ${BASE_SYS} \
        --lang_pairs ${LANG_PAIR} \
        --beam_size 5

evaluate_lang_direction "$TEST_DATASET" "$BASE_SYS" "$LANG_PAIR"
!
