#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=01-00:00:00

#SBATCH -o /home/dwu18/projects/llama-recipes/experiments/logs/out.comet.%j.o
#SBATCH -o /home/dwu18/projects/llama-recipes/experiments/logs/out.comet.%j.e

source activate llama_factory

export HF_HUB_CACHE=/scratch-shared/dwu18/cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

#comet-score -s en-zh/src -t en-zh/tgt --model Unbabel/wmt23-cometkiwi-da-xxl >> en-zh/score
#comet-score -s en-de/src -t en-de/tgt --model Unbabel/wmt23-cometkiwi-da-xxl >> en-de/score
#comet-score -s en-ru/src -t en-ru/tgt --model Unbabel/wmt23-cometkiwi-da-xxl >> en-ru/score

lang_pairs=("en-zh" "en-de" "en-ru")

for lang in "${lang_pairs[@]}"; do
    comet-score -s "$lang/src" -t "$lang/tgt" --model Unbabel/wmt23-cometkiwi-da-xxl >> "$lang/score"
done

