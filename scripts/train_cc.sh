#!/bin/bash

MODEL_TYPE=codellama # codebert, codegpt, lstm, code2vec, codeberta, graphcodebert, codellama
SHIFT_TYPE=different_time # different_project, different_author, different_time, case_study
RES_DIR=results/code_completion/$SHIFT_TYPE/$MODEL_TYPE

if [ ! -d $RES_DIR ]; then
  mkdir -p $RES_DIR
else
  echo dir exist
fi

DATA_DIR=dataset/code_completion/$SHIFT_TYPE
EPOCHS=20
BATCH=32
LR=2e-5
TRAIN_DATA=$DATA_DIR/train.tsv
VAL_DATA=$DATA_DIR/dev.tsv
TEST_DATA1=$DATA_DIR/test1.tsv
TEST_DATA2=$DATA_DIR/test2.tsv
TEST_DATA3=$DATA_DIR/test3.tsv

EMBEDDING_TYPE=1
EMBEDDING_DIM=128 # dimension of vectors (must be divisible by num_heads: 32 for codellama, 12 for codegpt)
EMBEDDING_PATH=/ # file for pre-trained vectors
EXPERIMENT_NAME=code_completion
EXPERIMENT_LOG=$RES_DIR/$EXPERIMENT_NAME'.txt'
echo $EXPERIMENT_NAME

CUDA_VISIBLE_DEVICES=0 python -B -m program_tasks.code_completion.main \
  --train_data $TRAIN_DATA --val_data $VAL_DATA \
  --test_data1 $TEST_DATA1 --test_data2 $TEST_DATA2 --test_data3 $TEST_DATA3 \
  --model_type $MODEL_TYPE \
  --ensemble_models 5 \
  --embedding_path $EMBEDDING_PATH \
  --embedding_type $EMBEDDING_TYPE \
  --embedding_dim $EMBEDDING_DIM \
  --do_train --do_eval \
  --epochs $EPOCHS --batch_size $BATCH --lr $LR --res_dir $RES_DIR \
  --experiment_name $EXPERIMENT_NAME | tee $EXPERIMENT_LOG

