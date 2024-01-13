#!/bin/bash

MODEL_TYPE=codellama # codeberta, code2vec, graphcodebert, codebert, codegpt, lstm
SHIFT_TYPE=different_time # different_project, different_author, different_time, case_study
RES_DIR=results/code_summary/$SHIFT_TYPE/$MODEL_TYPE

if [ ! -d $RES_DIR ]; then
  mkdir -p $RES_DIR
else
  echo dir exist
fi

DATA_DIR=dataset/code_summary/$SHIFT_TYPE
EPOCHS=20
BATCH=5
LR=1e-4
TK_PATH=$DATA_DIR/tk.pkl
TRAIN_DATA=$DATA_DIR/train.pkl # file for training dataset
VAL_DATA=$DATA_DIR/val.pkl # file for validation dataset
TEST_DATA1=$DATA_DIR/test1.pkl # file for test dataset1
TEST_DATA2=$DATA_DIR/test2.pkl # file for test dataset2
TEST_DATA3=$DATA_DIR/test3.pkl # file for test dataset3

EMBEDDING_TYPE=1
EMBEDDING_DIM=128 # dimension of embedding vectors
EMBEDDING_PATH=/ # file for pre-trained vectors
EXPERIMENT_NAME=code_summary
EXPERIMENT_LOG=$RES_DIR/$EXPERIMENT_NAME'.txt'
MAX_SIZE=20000 # number of training samples at each epoch

# echo $EXPERIMENT_NAME
CUDA_VISIBLE_DEVICES=5 python -B -m program_tasks.code_summary.main \
  --tk_path ${TK_PATH} --epochs ${EPOCHS} --batch ${BATCH} --lr ${LR} \
  --embed_dim ${EMBEDDING_DIM} --embed_path ${EMBEDDING_PATH} \
  --model_type ${MODEL_TYPE} \
  --train_data ${TRAIN_DATA} --val_data ${VAL_DATA} \
  --test_data1 ${TEST_DATA1} --test_data2 ${TEST_DATA2} --test_data3 ${TEST_DATA3} \
  --max_size ${MAX_SIZE} \
  --ensemble_models 5 \
  --do_train --do_eval \
  --embed_type ${EMBEDDING_TYPE} --experiment_name ${EXPERIMENT_NAME} \
  --res_dir ${RES_DIR} | tee $EXPERIMENT_LOG
