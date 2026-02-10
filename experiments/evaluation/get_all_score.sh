#!/bin/bash

# Language pairs
LANG_PAIRS=(en-zh zh-en en-de de-en en-ru ru-en en-cs cs-en en-is is-en)
LANG_PAIRS=(en-zh en-de en-ru en-cs en-is en-es en-uk en-ja en-hi)
LANG_PAIRS=(en-de en-fr en-nl en-it en-es en-pt en-ko en-ru en-zh en-uk en-ja en-cs en-is)
LANG_PAIRS=(en-de en-es en-ru en-zh en-fr en-nl en-it en-pt en-ko)

# Base directory
BASE_DIR="results/wmt24_testset"

# Define different settings
SETTINGS=(
  "ALMA-v1-7B-beam5"
  "ALMA-R-7B-beam5"
  "ALMA-v1-13B-beam5"
  "ALMA-R-13B-beam5"
)

SETTINGS=(
  "TowerInstruct-7B-v0.1-beam5"
  "TowerInstruct-7B-v0.2-beam5"
  "TowerInstruct-13B-v0.1-beam5"
  "TowerInstruct-Mistral-7B-v0.2-beam5"
)

SETTINGS=(
  "TowerInstruct-7B-v0.1-beam5"
  "TowerInstruct-13B-v0.1-beam5"
  "TowerInstruct-Mistral-7B-v0.2-beam5"
)


# Iterate over settings
for SETTING in "${SETTINGS[@]}"; do
  DIR="$BASE_DIR/$SETTING"

  # Initialize scores
  KIWI_SCORES=""
  COMET_SCORES=""
  SCORES=""

  # Loop through language pairs
  for PAIR in "${LANG_PAIRS[@]}"; do
    KIWI_FILE_NAME="$DIR/$PAIR/kiwi-xxl.score"
    KIWI_FILE_NAME="$DIR/$PAIR/comet.score"
    # COMET_FILE_NAME="$DIR/$PAIR/xcomet.score"

    # Read last line, extract final value, handle missing files
    KIWI_SCORE=$(tail -n 1 "$KIWI_FILE_NAME" 2>/dev/null | awk '{print $NF}')
    COMET_SCORE=$(tail -n 1 "$COMET_FILE_NAME" 2>/dev/null | awk '{print $NF}')

    # Convert to percentage and handle missing values
    KIWI_SCORE=$(echo "scale=2; ${KIWI_SCORE:-0} * 100" | bc)
    COMET_SCORE=$(echo "scale=2; ${COMET_SCORE:-0} * 100" | bc)

    KIWI_SCORES+="$KIWI_SCORE\t"
    COMET_SCORES+="$COMET_SCORE\t"
    SCORES+="$KIWI_SCORE\t$COMET_SCORE\t"
    # SCORES+="$KIWI_SCORE\t$COMET_SCORE\t"
  done

  # Trim trailing tab and print the formatted output
  SCORES=$(echo -e "$SCORES" | sed 's/\t$//')
  echo -e "$SETTING\t$SCORES"
done

