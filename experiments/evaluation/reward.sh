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
#SBATCH -e /home/dwu18/projects/llama-recipes/experiments/logs/out.score.%j.e

source activate py38cuda11
export HF_HUB_CACHE=/gpfs/work4/0/gus20642/dwu18/cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


SUBSET_NAME=$1
MODEL_NAME=$2
OUTPUT_DIR=$3
PRELOAD_PEFT_DIR=$4

if [ "$PRELOAD_PEFT_DIR" = "None" ] || [ -z "$PRELOAD_PEFT_DIR" ]; then
    echo $SUBSET_NAME $MODEL_NAME $OUTPUT_DIR
    python ../reward_inference.py --model_name $MODEL_NAME \
        --dataset da_dataset \
        --subset_name $SUBSET_NAME \
        --val_batch_size 8 \
        --do_sample False \
        --output_dir $OUTPUT_DIR \
        --lang_pairs en-de,en-ru,zh-en \
        --use_wandb
else
    python ../reward_inference.py --model_name $MODEL_NAME \
        --preload_peft_dir $PRELOAD_PEFT_DIR \
        --dataset da_dataset \
        --subset_name $SUBSET_NAME \
        --val_batch_size 8 \
        --do_sample False \
        --output_dir $OUTPUT_DIR \
        --lang_pairs en-de,en-ru,zh-en \
        --use_wandb
fi



:<<!
# 7B ALMA-v1, ALMA-R, Llama-2
python ../reward_inference.py --model_name haoranxu/ALMA-7B-Pretrain \
    --preload_peft_dir haoranxu/ALMA-7B-Pretrain-LoRA \
    --dataset da_dataset \
    --subset_name wmt-qe-2022.train.csv \
    --val_batch_size 32 \
    --do_sample False \
    --output_dir scores/ALMA-v1/wmt-qe-22-train \
    --lang_pairs en-de,en-ru,zh-en \
    --use_wandb


python ../reward_inference.py --model_name haoranxu/ALMA-7B-R \
    --dataset da_dataset \
    --subset_name wmt-qe-2022.train.csv \
    --val_batch_size 32 \
    --do_sample False \
    --output_dir scores/ALMA-r/wmt-qe-22-train \
    --lang_pairs en-de,en-ru,zh-en \
    --use_wandb

python ../reward_inference.py --model_name meta-llama/Llama-2-7b-hf \
    --dataset da_dataset \
    --subset_name wmt-qe-2022.train.csv \
    --val_batch_size 32 \
    --do_sample False \
    --output_dir scores/Llama2-7B/wmt-qe-22-train \
    --lang_pairs en-de,en-ru,zh-en \
    --use_wandb

python ../reward_inference.py --model_name meta-llama/Meta-Llama-3-8B \
    --dataset da_dataset \
    --subset_name wmt-qe-2022.train.csv \
    --val_batch_size 32 \
    --do_sample False \
    --output_dir scores/Llama3-8B/wmt-qe-22-train \
    --lang_pairs en-de,en-ru,zh-en \
    --use_wandb

# 13B ALMA-v1, ALMA-R, Llama-2
python ../reward_inference.py --model_name haoranxu/ALMA-13B-Pretrain \
    --preload_peft_dir haoranxu/ALMA-13B-Pretrain-LoRA \
    --dataset da_dataset \
    --subset_name wmt-qe-2022.train.csv \
    --val_batch_size 8 \
    --do_sample False \
    --output_dir scores/ALMA-v1-13B/wmt-qe-22-train \
    --lang_pairs en-de,en-ru,zh-en \
    --use_wandb


python ../reward_inference.py --model_name haoranxu/ALMA-13B-R \
    --dataset da_dataset \
    --subset_name wmt-qe-2022.train.csv \
    --val_batch_size 8 \
    --do_sample False \
    --output_dir scores/ALMA-r-13B/wmt-qe-22-train \
    --lang_pairs en-de,en-ru,zh-en \
    --use_wandb

python ../reward_inference.py --model_name meta-llama/Llama-2-13b-hf \
    --dataset da_dataset \
    --subset_name wmt-qe-2022.train.csv \
    --val_batch_size 8 \
    --do_sample False \
    --output_dir scores/Llama2-13B/wmt-qe-22-train \
    --lang_pairs en-de,en-ru,zh-en \
    --use_wandb
!
