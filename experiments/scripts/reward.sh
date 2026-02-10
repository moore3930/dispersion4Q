#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=01-00:00:00

#SBATCH -o /home/dwu18/projects/llama-recipes/experiments/logs/out.score.%j.o
#SBATCH -o /home/dwu18/projects/llama-recipes/experiments/logs/out.score.%j.e

source activate py38cuda11
export HF_HUB_CACHE=/gpfs/work4/0/gus20642/dwu18/cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

ALPHA=$1
BETA=$2
GAMA=$3
LR=$4

echo "ALPHA is set to: $ALPHA"
echo "BETA is set to: $BETA"
echo "GAMA is set to: $GAMA"
echo "LR is set to: $LR"

SETTING=${ALPHA}-${BETA}-${GAMA}-${LR}-full

# Get scores (log_probs) on WMT22 Metric Data (Train & Test Set)
for EPOCH in 0 1; do
    # Inference ALMA
    python reward_inference.py --model_name haoranxu/ALMA-7B-Pretrain \
        --preload_peft_dir haoranxu/ALMA-7B-Pretrain-LoRA \
        --peft_model ./checkpoints/7B/calibration/gpt-4o-mini/${SETTING}/${EPOCH} \
        --dataset da_dataset \
        --subset_name wmt-qe-2022.train.csv \
        --val_batch_size 32 \
        --do_sample False \
        --output_dir scores/calibration/${SETTING}/${EPOCH}/wmt-qe-22-train \
        --xpo_hyper ${XPO_HYPER} \
        --alpha ${ALPHA} \
        --beta ${BETA} \
        --lang_pairs en-de,en-ru \
        --use_wandb
done
