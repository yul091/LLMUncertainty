#!/bin/bash

MODULE_ID=0 # 0 is code summary
# MODEL_TYPE=codellama # lstm, codebert, codegpt, code2vec, codeberta, graphcodebert, codellama
for MODEL_TYPE in code2vec codebert codegpt; do
  SHIFT_TYPE=different_time # different_time, different_project, different_author
  DATA_DIR=dataset/code_summary/$SHIFT_TYPE
  RES_DIR=results/code_summary/$SHIFT_TYPE/$MODEL_TYPE
  SAVE_DIR=Uncertainty_Results/$SHIFT_TYPE/$MODEL_TYPE
  TRAIN_BATCH_SIZE=5
  TEST_BATCH_SIZE=5
  LOG_DIR=Uncertainty_logging/code_summary/$SHIFT_TYPE
  EXPERIMENT_LOG=$LOG_DIR/$MODEL_TYPE'.txt'
  for dir in "${RES_DIR}"/ensemble_model-*; do
      # Check if it's indeed a directory
      # if [ -d "$dir" ]; then
      if [ -d "$dir" ] && [ "$(basename "$dir")" != "ensemble_model-0" ]; then
          # Append to the ENSEMBLE_DIRS string, separated by space
          ENSEMBLE_DIRS="${ENSEMBLE_DIRS} ${dir}"
      fi
  done

  if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
  else
    echo dir exist
  fi

  if [ ! -d $SAVE_DIR ]; then
    mkdir -p $SAVE_DIR
  else
    echo dir exist
  fi

  CUDA_VISIBLE_DEVICES=6 python test_uncertainty.py \
    --module_id $MODULE_ID --res_dir $RES_DIR/ensemble_model-0 \
    --data_dir $DATA_DIR --save_dir $SAVE_DIR \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --test_batch_size $TEST_BATCH_SIZE \
    --ensemble_dirs $ENSEMBLE_DIRS | tee $EXPERIMENT_LOG
done

