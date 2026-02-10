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

comet-score -s src -t ref --model Unbabel/wmt23-cometkiwi-da-xxl >> score


