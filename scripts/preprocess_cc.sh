# PROJECT=different_time # different_author, different_project, different_time, case_study

for SHIFT in different_time different_project; do
    if [ $SHIFT != "case_study" ]; then
        DATA_DIR=data/main/$SHIFT
    elif [ $SHIFT == "case_study" ]; then
        DATA_DIR=data/case_study
    fi
    DEST_DIR=dataset/code_completion/$SHIFT

    python -B -m program_tasks.code_completion.prepro \
        --data_dir $DATA_DIR \
        --dest_dir $DEST_DIR
done