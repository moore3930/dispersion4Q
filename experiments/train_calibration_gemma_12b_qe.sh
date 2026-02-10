#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -p gpu
#SBATCH --gres gpu:4
#SBATCH --partition=gpu_h100
#SBATCH --time=01-00:00:00

#SBATCH -o /gpfs/work4/0/gus20642/dwu18/log/out.calibration.%j.o
#SBATCH -o /gpfs/work4/0/gus20642/dwu18/log/out.calibration.%j.e

source activate test_time_mt

export HF_HUB_CACHE=/gpfs/work4/0/gus20642/dwu18/cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


################## MAIN ##################

ALPHA=$1
BETA=$2
GAMA=$3
LR=$4
SUBSET=$5
LIST_SIEZ=$6
METRIC=$7
BASE_MODEL=$8
MASTER_PORT=$9

echo "ALPHA is set to: $ALPHA"
echo "BETA is set to: $BETA"
echo "GAMA is set to: $GAMA"
echo "LR is set to: $LR"
echo "Subset is set to: $SUBSET"
echo "List size is set to: $LIST_SIEZ"
echo "Base_model is set to: $BASE_MODEL"

# final_loss = alpha * chose_nll_acc_loss + beta * value_acc_loss + gama * cpo_acc_loss
SETTING=${ALPHA}-${BETA}-${GAMA}-${LR}-${METRIC}-qe25
CKP_DIR=/gpfs/work4/0/gus20642/dwu18/calibration/checkpoints

echo "CKP: $CKP_DIR/$BASE_MODEL/calibration_v2/${SUBSET}/${SETTING}"

# Train
LANG_PAIRS="cs-uk,cs-de,ja-zh,en-ar,en-zh,en-cs,en-et,en-is,en-ja,en-ko,en-ru,en-sr,en-uk,en-bh"
CUDA_LAUNCH_BLOCKING=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port $MASTER_PORT -m llama_recipes.calibration --enable_fsdp --use_peft --peft_method lora \
        --model_name google/$BASE_MODEL \
        --output_dir $CKP_DIR/$BASE_MODEL/calibration_v2/${SUBSET}/${SETTING} \
        --dataset calibration_v2 \
        --subset_name ${SUBSET} \
        --metric ${METRIC} \
        --batching_strategy padding \
        --num_epochs 1 \
        --lr $LR \
        --batch_size_training 16 \
        --val_batch_size 16 \
        --gradient_accumulation_steps 16 \
        --alpha $ALPHA \
        --beta $BETA \
	--gama $GAMA \
        --lang_pairs $LANG_PAIRS \
        --listwise_loss \
        --list_size $LIST_SIEZ \
        --use_wandb

