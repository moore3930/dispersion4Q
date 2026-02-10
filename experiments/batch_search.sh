# sbatch train_tower_calibration_7B.sh 1.0 1.0 0.0 5e-5 gpt-4o-mini-16-1.0-98-new 16 kiwi-xxl.score TowerInstruct-Mistral-7B-v0.2


:<<!
# Subset
LANG_PAIRS="en-de,en-fr,en-nl,en-it,en-es,en-pt,en-ko,en-ru,en-zh"

# Convert comma-separated string to array
IFS=',' read -ra PAIRS <<< "$LANG_PAIRS"

for LANG_PAIR in "${PAIRS[@]}"; do
    echo "Running inference for $LANG_PAIR"
    sbatch ./scripts/inference_lp.sh 1.0 1.0 0.0 5e-5 gpt-4o-mini-16-1.0-98-new 16 kiwi-xxl.score TowerInstruct-Mistral-7B-v0.2 "$LANG_PAIR"
done
!

# Paragraph-level evaluation WMT24++
LANG_PAIRS="en-zh,en-de,en-fr,en-ru"

# Zhu
LANG_PAIRS="en-zh,zh-en,en-de,de-en"

:<<!
# Convert comma-separated string to array
IFS=',' read -ra PAIRS <<< "$LANG_PAIRS"

for LANG_PAIR in "${PAIRS[@]}"; do
    echo "Running inference for $LANG_PAIR"
    sbatch ./scripts/inference_lp.sh 1.0 0.0 1.0 5e-5 rebuttal 16 kiwi-xxl.score TowerInstruct-Mistral-7B-v0.2 "$LANG_PAIR"
    sbatch ./scripts/inference_lp.sh 1.0 0.0 0.0 5e-5 rebuttal 16 kiwi-xxl.score TowerInstruct-Mistral-7B-v0.2 "$LANG_PAIR"
    sbatch ./scripts/inference_lp.sh 1.0 0.0 1.0 5e-5 rebuttal 16 kiwi-xxl.score TowerInstruct-Mistral-7B-v0.2 "$LANG_PAIR"
done
!


LANG_PAIRS="zh-en,en-de,de-en"
# Convert comma-separated string to array
IFS=',' read -ra PAIRS <<< "$LANG_PAIRS"

for LANG_PAIR in "${PAIRS[@]}"; do
    echo "Running inference for $LANG_PAIR"
    sbatch ./scripts/inference_lp.sh 1.0 1.0 0.0 5e-5 rebuttal 16 kiwi-xxl.score Mistral-7B-Instruct-v0.3 "$LANG_PAIR"
done
