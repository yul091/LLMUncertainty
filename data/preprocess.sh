
for DIR in main/different_time main/different_project main/different_author; do
    BASENAME=$(basename $DIR)
    OUT=preprocessed/$BASENAME
    python preprocess.py \
        --base_dir $DIR \
        --output_dir $OUT \
        --train_file train.txt \
        --dev_file dev.txt \
        --test1_file test1.txt \
        --test2_file test2.txt \
        --test3_file test3.txt 
done


DIR=case_study
OUT=preprocessed/$DIR
python preprocess.py \
    --base_dir $DIR \
    --output_dir $OUT \
    --train_file train.txt \
    --dev_file dev.txt \
    --test_file test.txt
