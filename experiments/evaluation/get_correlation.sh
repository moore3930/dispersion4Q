#!/bin/bash

#!/bin/bash

# List of base directories
BASE_DIRS=(
    "scores/Llama2-7B"
    "scores/Llama2-13B"
    "scores/ALMA-Base-7B"
    "scores/ALMA-Base-13B"
    "scores/ALMA-v1-7B"
    "scores/ALMA-v1-13B"
    "scores/Tower-base-7B"
    "scores/Tower-base-13B"
    "scores/Tower-v1-7B"
    "scores/Tower-v1-13B"
    "scores/ALMA-r-7B"
    "scores/ALMA-r-13B"
)

BASE_DIRS=(
    "scores/Llama2-7B"
    "scores/Llama2-13B"
    "scores/Tower-base-7B"
    "scores/Tower-base-13B"
    "scores/Tower-v1-7B"
    "scores/Tower-v1-13B"
)

X=-2
METHOD=spearman
METHOD=kendall
TEST_DATA=wmt-qe-22-test
# TEST_DATA=wmt-qe-22-train

SETTINGS=(
    "1.0-0.0-0.0-5e-5"
)

LANG_PAIRS=("en-de" "en-ru")

for BASE_DIR in "${BASE_DIRS[@]}"; do
    for SETTING in "${SETTINGS[@]}"; do
        results=()
        for LANG_PAIR in "${LANG_PAIRS[@]}"; do
            SRC_LANG=$(echo "$LANG_PAIR" | cut -d'-' -f1)
            TGT_LANG=$(echo "$LANG_PAIR" | cut -d'-' -f2)
            FILE_PATH="$BASE_DIR/${TEST_DATA}/${LANG_PAIR}/score.${LANG_PAIR}.${TGT_LANG}"

            if [ "$METHOD" = "pearson" ]; then
                result=$(python ./sperson.py --file_name "$FILE_PATH" --column_x $X --method $METHOD | grep -oP "Pearson's correlation coefficient: \K[0-9.-]+")
            elif [ "$METHOD" = "kendall" ]; then
                result=$(python ./sperson.py --file_name "$FILE_PATH" --column_x $X --method $METHOD | grep -oP "kendall's rank correlation coefficient: \K[0-9.-]+")
            else
                result=$(python ./sperson.py --file_name "$FILE_PATH" --column_x $X | grep -oP "Spearman's rank correlation coefficient: \K[0-9.-]+")
            fi
            results+=("$result")
        done
        echo "${METHOD}:    ${results[0]}   ${results[1]}   ${results[2]}"
    done
done



:<<!

# Base directory
BASE_DIR="scores/Llama2-7B"
BASE_DIR="scores/Llama2-13B"
BASE_DIR="scores/ALMA-v1-7B"
BASE_DIR="scores/ALMA-v1-13B"
BASE_DIR="scores/ALMA-r-7B"
BASE_DIR="scores/ALMA-r-13B"

X=-2
METHOD=speaman
TEST_DATA=wmt-qe-22-test

SETTINGS=(
    "1.0-0.0-0.0-5e-5"
)

LANG_PAIRS=("en-de" "en-ru")

for SETTING in "${SETTINGS[@]}"; do
    results=()
    for LANG_PAIR in "${LANG_PAIRS[@]}"; do
        SRC_LANG=$(echo "$LANG_PAIR" | cut -d'-' -f1)
        TGT_LANG=$(echo "$LANG_PAIR" | cut -d'-' -f2)
        FILE_PATH="$BASE_DIR/${TEST_DATA}/${LANG_PAIR}/score.${LANG_PAIR}.${TGT_LANG}"

        if [ "$METHOD" = "pearson" ]; then
            result=$(python ./sperson.py --file_name "$FILE_PATH" --column_x $X --method $METHOD | grep -oP "Pearson's correlation coefficient: \K[0-9.-]+")
        else
            result=$(python ./sperson.py --file_name "$FILE_PATH" --column_x $X | grep -oP "Spearman's rank correlation coefficient: \K[0-9.-]+")
        fi
        results+=("$result")
    done
    echo "${METHOD}:	${results[0]}	${results[1]}	${results[2]}"
done
!
