# Tower-7B-Mistral with KIWI-XXL (vLLM inference, AWQ config)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
sbatch train_dispersion.sh 5e-5 TowerInstruct-Mistral-7B-v0.2 1 vllm experiments/configs/vllm_awq.json
