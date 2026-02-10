#!/bin/bash

# Language pairs

# Tower
LANG_PAIRS=(en-de en-es en-ru en-zh en-fr en-nl en-it en-pt en-ko)
# LANG_PAIRS=(en-de en-fr en-zh en-ru)
LANG_PAIRS=(en-zh zh-en en-de de-en)


# ALMA
#LANG_PAIRS=(en-de en-cs en-is en-zh en-ru)
#LANG_PAIRS=(de-en cs-en is-en zh-en ru-en)

# Base directory
BASE_DIR="../results/TowerBase-7B-v0.1/calibration/wmt24_testset/gpt-4o-mini-16-1.0-98-new"
BASE_DIR="../results/TowerInstruct-7B-v0.2/calibration/wmt24_testset/gpt-4o-mini-16-1.0-98-new"

BASE_DIR="../results/TowerBase-7B-v0.1/calibration/wmt24_testset/gpt-4o-mini-16-1.0-98-new"
BASE_DIR="../results/TowerBase-13B-v0.1/calibration/wmt24_testset/gpt-4o-mini-16-1.0-98-new"
BASE_DIR="../results/TowerInstruct-Mistral-7B-v0.2/calibration/wmt24_testset/gpt-4o-mini-16-1.0-98-new"
BASE_DIR="../results/TowerInstruct-Mistral-7B-v0.2/calibration/wmt24_testset/gpt-4o-mini-16-1.0-98-new"

# BASE_DIR="../results/TowerInstruct-Mistral-7B-v0.2/calibration/wmt24_plus_doc_testset/gpt-4o-mini-16-1.0-98-new"
BASE_DIR="../results/TowerInstruct-Mistral-7B-v0.2/calibration/wmt22_testset/rebuttal"
BASE_DIR="../results/TowerInstruct-Mistral-7B-v0.2/calibration/wmt22_testset/rebuttal"


# Define different settings

SETTINGS=(
  "1.0-0.0-0.0-5e-5-llama3-beam5"
  "1.0-0.0-1.0-5e-5-llama3-beam5"
  "1.0-1.0-0.0-5e-5-llama3-beam5"
  "1.0-0.0-0.0-1e-4-llama3-beam5"
  "1.0-0.0-1.0-1e-4-llama3-beam5"
  "1.0-1.0-0.0-1e-4-llama3-beam5"
)

SETTINGS=(
  "1.0-0.0-0.0-5e-5-kiwi-xxl.score-beam5"
  "1.0-0.0-1.0-5e-5-kiwi-xxl.score-beam5"
  "1.0-1.0-0.0-5e-5-kiwi-xxl.score-beam5"
)


# Iterate over settings
for SETTING in "${SETTINGS[@]}"; do
  DIR="$BASE_DIR/$SETTING/0"

  # Initialize scores
  KIWI_SCORES=""
  COMET_SCORES=""
  XCOMET_SCORES=""
  SCORES=""

  # Loop through language pairs
  for PAIR in "${LANG_PAIRS[@]}"; do
    KIWI_FILE_NAME="$DIR/$PAIR/kiwi-xl.score"
    COMET_FILE_NAME="$DIR/$PAIR/comet.score"
    XCOMET_FILE_NAME="$DIR/$PAIR/xcomet.qe.score"

    # Read last line, extract final value, handle missing files
    KIWI_SCORE=$(tail -n 1 "$KIWI_FILE_NAME" 2>/dev/null | awk '{print $NF}')
    COMET_SCORE=$(tail -n 1 "$COMET_FILE_NAME" 2>/dev/null | awk '{print $NF}')
    XCOMET_SCORE=$(tail -n 1 "$XCOMET_FILE_NAME" 2>/dev/null | awk '{print $NF}')

    # Convert to percentage and handle missing values
    KIWI_SCORE=$(echo "scale=2; ${KIWI_SCORE:-0} * 100" | bc)
    COMET_SCORE=$(echo "scale=2; ${COMET_SCORE:-0} * 100" | bc)
    XCOMET_SCORE=$(echo "scale=2; ${XCOMET_SCORE:-0} * 100" | bc)

    KIWI_SCORES+="$KIWI_SCORE\t"
    COMET_SCORES+="$COMET_SCORE\t"
    XCOMET_SCORES+="$XCOMET_SCORE\t"
    # SCORES+="$COMET_SCORE\t$KIWI_SCORE\t$XCOMET_SCORE\t"
    # SCORES+="$COMET_SCORE\t"
    SCORES+="$KIWI_SCORE\t"
  done

  # Trim trailing tab and print the formatted output
  SCORES=$(echo -e "$SCORES" | sed 's/\t$//')
  echo -e "$SETTING\t$SCORES"
done

