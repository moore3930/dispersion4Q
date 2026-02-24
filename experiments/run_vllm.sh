# Tower-7B-Mistral with KIWI-XXL (vLLM inference)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
LANG_SET_ARG="${1:-${LANG_SET:-tower1}}"
echo "Submitting run_vllm with LANG_SET=${LANG_SET_ARG}"
sbatch --export=ALL,LANG_SET="${LANG_SET_ARG}" train_dispersion.sh 5e-5 TowerInstruct-Mistral-7B-v0.2 1 vllm
