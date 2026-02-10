#!/bin/bash

# Arrays
VPO_HYP=(20 21 22)
ALPHA=(0 0.1)
BETA=(1 1.5 2)
LR=(1e-6 2e-6)
LANG_PAIRS=(en-zh en-de)

# Base directory
BASE_DIR="results_small"

# Loop through all combinations
for vpo in "${VPO_HYP[@]}"; do
  for alpha in "${ALPHA[@]}"; do
    for beta in "${BETA[@]}"; do
      for lr in "${LR[@]}"; do
        # Construct the directory path
        DIR="$BASE_DIR/ALMA-VPO-$vpo-$alpha-$beta-$lr"
        echo "$vpo $alpha $beta $lr"

        # Loop language pairs
        PAIRS=""
        KIWI_SCORES=""
        COMET_SCORES=""
        for PAIR in "${LANG_PAIRS[@]}"; do
          KIWI_FILE_NAME="$DIR/$PAIR/kiwi.score"
          KIWI_SCORE=$(tail -n 1 "$KIWI_FILE_NAME" | awk '{print $NF}')

          COMET_FILE_NAME="$DIR/$PAIR/comet.score"
          COMET_SCORE=$(tail -n 1 "$COMET_FILE_NAME" | awk '{print $NF}')

          KIWI_SCORES="$KIWI_SCORES $KIWI_SCORE"
          COMET_SCORES="$COMET_SCORES $COMET_SCORE"
          PAIRS="$PAIRS $PAIR"
        done
        echo $PAIRS
        echo $KIWI_SCORES
        echo $COMET_SCORES
      done
    done
  done
done


