#!/bin/bash

# Base directory
BASE_DIR="/home/dwu18/projects/value_finetuning/experiments/scores/calibration"
BASE_DIR="../scores/7B/calibration/gpt-4o-mini-16-1.0-98-ensemble"
# BASE_DIR="../scores/7B/calibration/on-policy-alma-v1/"
BASE_DIR="../scores/tower-7B/calibration/gpt-4o-mini-16-1.0-98"
BASE_DIR="../scores/tower-13B/calibration/gpt-4o-mini-16-1.0-98"
Base_dir="../scores/TowerBase-13B-v0.1/calibration/gpt-4o-mini-16-1.0-98-new"
Base_dir="../scores/TowerBase-7B-v0.1/calibration/gpt-4o-mini-16-1.0-98-new"

X=-2
# METHOD=pearson
METHOD=kendall
TEST_DATA=wmt-qe-22-test
# TEST_DATA=wmt-qe-22-train

# Different settings
SETTINGS=(
    "1.0-0.0-0.0-5e-5-llama3"
    "1.0-0.0-1.0-5e-5-llama3"
    "1.0-1.0-0.0-5e-5-llama3"
    "1.0-0.0-0.0-1e-4-llama3"
    "1.0-0.0-1.0-1e-4-llama3"
    "1.0-1.0-0.0-1e-4-llama3"
)

SETTINGS=(
    "1.0-0.0-0.0-5e-5"
    "1.0-0.0-1.0-5e-5"
    "1.0-1.0-0.0-5e-5"
    "1.0-0.0-0.0-1e-4"
    "1.0-0.0-1.0-1e-4"
    "1.0-1.0-0.0-1e-4"
)

SETTINGS=(
    "1.0-1.0-0.0-5e-5-kiwi-xxl.score"
    "1.0-0.0-0.0-5e-5-kiwi-xxl.score"
    "1.0-0.0-1.0-5e-5-kiwi-xxl.score"
)


EPOCH=0
LANG_PAIRS=("en-de" "en-ru" "zh-en")
LANG_PAIRS=("en-de" "en-ru")

for SETTING in "${SETTINGS[@]}"; do
    results=()
    for LANG_PAIR in "${LANG_PAIRS[@]}"; do
        SRC_LANG=$(echo "$LANG_PAIR" | cut -d'-' -f1)
        TGT_LANG=$(echo "$LANG_PAIR" | cut -d'-' -f2)
        FILE_PATH="$BASE_DIR/${SETTING}/${EPOCH}/${TEST_DATA}/${LANG_PAIR}/score.${LANG_PAIR}.${TGT_LANG}"

        if [ "$METHOD" = "pearson" ]; then
            result=$(python ./sperson.py --file_name "$FILE_PATH" --column_x $X --method $METHOD | grep -oP "Pearson's correlation coefficient: \K[0-9.-]+")
        elif [ "$METHOD" = "kendall" ]; then
            result=$(python ./sperson.py --file_name "$FILE_PATH" --column_x $X --method $METHOD | grep -oP "kendall's rank correlation coefficient: \K[0-9.-]+")
        else
            result=$(python ./sperson.py --file_name "$FILE_PATH" --column_x $X | grep -oP "Spearman's rank correlation coefficient: \K[0-9.-]+")
        fi
        results+=("$result")
    done
    echo "${SETTING}-${EPOCH}-${METHOD}:	${results[0]}	${results[1]}	${results[2]}"
done

