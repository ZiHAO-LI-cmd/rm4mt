#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=/scratch/project_462000941/members/zihao/rm4mt/logs/eval_local/%x_%j.out
#SBATCH --error=/scratch/project_462000941/members/zihao/rm4mt/logs/eval_local/%x_%j.err
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --account=project_462000964

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/rm4mt_env/bin/activate

INPUT_DIR=""
MODEL_NAME=""
THINKING_BUDGET=
ENABLE_WAIT_INSERTION=""
SEED=42
ADD_DOC_FOR_RAGTRANS=""

if [ "$THINKING_BUDGET" -eq 0 ]; then
    TEMPERATURE=0.7
    TOP_P=0.8
else
    TEMPERATURE=0.6
    TOP_P=0.95
fi

DATASET_NAME=$(basename "$INPUT_DIR")

if [ "$ENABLE_WAIT_INSERTION" = "False" ]; then
    if [ "$DATASET_NAME" == "RAGtrans" ] && [ "$ADD_DOC_FOR_RAGTRANS" = "False" ]; then
        OUTPUT_DIR="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated/${DATASET_NAME}_without_doc/$(basename ${MODEL_NAME})/budget_${THINKING_BUDGET}"
    else
        OUTPUT_DIR="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated/${DATASET_NAME}/$(basename ${MODEL_NAME})/budget_${THINKING_BUDGET}"
    fi
else
    OUTPUT_DIR="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_wait_translated/${DATASET_NAME}/$(basename ${MODEL_NAME})/budget_${THINKING_BUDGET}"
fi

SCRIPT="eval_qwen.py"

echo "Translating files in $INPUT_DIR ..."

CMD="python $SCRIPT \
    --input_dir \"$INPUT_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --model \"$MODEL_NAME\" \
    --temperature \"$TEMPERATURE\" \
    --top_p \"$TOP_P\" \
    --max_new_tokens 12000 \
    --thinking_budget \"$THINKING_BUDGET\" \
    --seed \"$SEED\" \
    --device_map \"auto\""

if [ "$ADD_DOC_FOR_RAGTRANS" = "True" ]; then
    CMD="$CMD --add_doc_for_ragtrans"
fi

if [ "$ENABLE_WAIT_INSERTION" = "True" ]; then
    CMD="$CMD --enable_wait_insertion"
fi

eval $CMD

echo "Done: results saved to $OUTPUT_DIR"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
