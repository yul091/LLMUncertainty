
# SHIFT=case_study # different_project, different_time, different_author
TASK=code_completion # code_summary, code_completion
MODEL=codellama # code2vec, coderoberta, graphcodebert, lstm, codebert, codegpt

DATADIR=dataset
MODELDIR=results
METRICDIR=Uncertainty_Results
OUTDIR=Uncertainty_Eval/input_validation

for SHIFT in different_author different_time different_project case_study; do
    CUDA_VISIBLE_DEVICES=7 python input_validation.py \
        --shift_type $SHIFT \
        --task $TASK \
        --model $MODEL \
        --uncertainty_dir $METRICDIR \
        --out_dir $OUTDIR \
        --strategy coverage
done