#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
##SBATCH --time=01-00:00:00
#SBATCH --time=00-2:00:00

#SBATCH -o /gpfs/work4/0/gus20642/dwu18/log/out.inference.%j.o
#SBATCH -e /gpfs/work4/0/gus20642/dwu18/log/out.inference.%j.e

source activate py38cuda11

export HF_HUB_CACHE=/gpfs/work4/0/gus20642/dwu18/cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

evaluate_lang_directions() {
    # Parameters
    local TEST_DATASET="$1"  # Test dataset name
    local BASE_SYS="$2"     # Base system directory
    local LANG_PAIR="$3"     # Single language direction, e.g., "en-de"

    # Extract source and target language codes
    local SRC_LANG=$(echo $LANG_PAIR | cut -d'-' -f1)
    local TGT_LANG=$(echo $LANG_PAIR | cut -d'-' -f2)

    # Define base source and target directories
    local BASE_SRC="/gpfs/work4/0/gus20642/dwu18/project/calibrating-llm-mt/src/llama_recipes/customer_data/${TEST_DATASET}/test"
    local BASE_TGT="/gpfs/work4/0/gus20642/dwu18/project/calibrating-llm-mt/src/llama_recipes/customer_data/${TEST_DATASET}/test"
 
    # Define the file paths
    local SRC_FILE="${BASE_SRC}/${LANG_PAIR}/test.${LANG_PAIR}.${SRC_LANG}"
    local TGT_FILE="${BASE_TGT}/${LANG_PAIR}/test.${LANG_PAIR}.${TGT_LANG}"
    local SYS_FILE="${BASE_SYS}/${LANG_PAIR}/hyp.${LANG_PAIR}.${TGT_LANG}"

    # Define the output score files
    local COMET_SCORE_FILE="./${BASE_SYS}/${LANG_PAIR}/comet.score"
    local XCOMET_SCORE_FILE="./${BASE_SYS}/${LANG_PAIR}/xcomet.score"
    local XCOMET_XL_SCORE_FILE="./${BASE_SYS}/${LANG_PAIR}/xcomet-xl.score"
    local KIWI_SCORE_FILE="./${BASE_SYS}/${LANG_PAIR}/kiwi.score"
    local KIWI_XL_SCORE_FILE="./${BASE_SYS}/${LANG_PAIR}/kiwi-xl.score"
    local KIWI_XXL_SCORE_FILE="./${BASE_SYS}/${LANG_PAIR}/kiwi-xxl.score" 

    echo "Calculating COMET scores for ${LANG_PAIR}..."

    # Run COMET scoring
    comet-score -s $SRC_FILE -t $SYS_FILE -r $TGT_FILE --model Unbabel/wmt22-comet-da >> $COMET_SCORE_FILE
    # comet-score -s $SRC_FILE -t $SYS_FILE --model Unbabel/XCOMET-XXL >> $XCOMET_SCORE_FILE
    # comet-score -s $SRC_FILE -t $SYS_FILE --model Unbabel/XCOMET-XL  >> $XCOMET_XL_SCORE_FILE
    comet-score -s $SRC_FILE -t $SYS_FILE --model Unbabel/wmt23-cometkiwi-da-xl >> $KIWI_XL_SCORE_FILE
    comet-score -s $SRC_FILE -t $SYS_FILE --model Unbabel/wmt23-cometkiwi-da-xxl >> $KIWI_XXL_SCORE_FILE

    echo "Finished ${LANG_PAIR}"
}



################## MAIN ##################

ALPHA=$1
BETA=$2
GAMA=$3
LR=$4
SUBSET=$5
LIST_SIEZ=$6
METRIC=$7
BASE_MODEL=$8
LP=$9

echo "ALPHA is set to: $ALPHA"
echo "BETA is set to: $BETA"
echo "GAMA is set to: $GAMA"
echo "LR is set to: $LR"
echo "Subset is set to: $SUBSET"
echo "List size is set to: $LIST_SIEZ"
echo "Base_model is set to: $BASE_MODEL"

# final_loss = alpha * chose_nll_acc_loss + beta * value_acc_loss + gama * cpo_acc_loss
SETTING=${ALPHA}-${BETA}-${GAMA}-${LR}-${METRIC}
TEST_DATASET=wmt22_testset
# TEST_DATASET=wmt24_plus_doc_testset
CKP_DIR=/gpfs/work4/0/gus20642/dwu18/project/calibrating-llm-mt/experiments/checkpoints
# CKP_DIR=/gpfs/work4/0/gus20642/dwu18/calibration/checkpoints

echo "CKP: $CKP_DIR/$BASE_MODEL/calibration/${SUBSET}/${SETTING}"
echo "RESULTS: results/$BASE_MODEL/calibration/${TEST_DATASET}/${SUBSET}/${SETTING}-beam5"

# Test
for EPOCH in 0; do
    BASE_SYS=results/$BASE_MODEL/calibration/${TEST_DATASET}/${SUBSET}/${SETTING}-beam5/${EPOCH}
    python inference_formal.py --model_name mistralai/$BASE_MODEL \
            --peft_model $CKP_DIR/$BASE_MODEL/calibration/${SUBSET}/${SETTING}/${EPOCH} \
            --dataset ${TEST_DATASET} \
            --val_batch_size 8 \
            --do_sample False \
            --output_dir ${BASE_SYS} \
            --lang_pairs ${LP} \
            --beam_size 5
    evaluate_lang_directions ${TEST_DATASET} ${BASE_SYS} ${LP}
done

