#!/bin/bash

# Base directory
BASE_DIR="../scores/7B/calibration/gpt-4o-mini-16-1.25-98"
X=-2
METHOD=pearson
METHOD=spearman  # 注意这里应使用正确的方法名

# Different settings
SETTINGS=(
    "1.0-0.0-0.0-5e-5"
    "1.0-0.0-1.0-5e-5"
    "1.0-1.0-0.0-5e-5"
    "1.0-0.0-0.0-1e-4"
    "1.0-0.0-1.0-1e-4"
    "1.0-1.0-0.0-1e-4"
)

EPOCH=0
LANG_PAIRS=("en-de" "en-ru" "zh-en")

for SETTING in "${SETTINGS[@]}"; do
    output="${SETTING}-${EPOCH}-${METHOD}"  # 先拼接前缀
    for LANG_PAIR in "${LANG_PAIRS[@]}"; do
        SRC_LANG=$(echo "$LANG_PAIR" | cut -d'-' -f1)
        TGT_LANG=$(echo "$LANG_PAIR" | cut -d'-' -f2)
        FILE_PATH="$BASE_DIR/${SETTING}/${EPOCH}/wmt-qe-22-test/${LANG_PAIR}/score.${LANG_PAIR}.${TGT_LANG}"

        if [ "$METHOD" = "pearson" ]; then
            result=$(python ./sperson.py --file_name "$FILE_PATH" --column_x $X --method $METHOD | grep -oP "Pearson's correlation coefficient: \K[0-9.-]+")
        else
            result=$(python ./sperson.py --file_name "$FILE_PATH" --column_x $X | grep -oP "Spearman's rank correlation coefficient: \K[0-9.-]+")
        fi

        output+="\t$result"  # 结果用 \t 连接
    done
    echo -e "$output"  # 用 -e 解析 \t
done

