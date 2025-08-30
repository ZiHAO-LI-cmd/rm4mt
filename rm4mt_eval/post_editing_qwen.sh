#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=/scratch/project_462000941/members/zihao/rm4mt/logs/post-editing/%x_%j.out
#SBATCH --error=/scratch/project_462000941/members/zihao/rm4mt/logs/post-editing/%x_%j.err
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --account=project_462000941

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/rm4mt_env/bin/activate

INPUT_DIR=""
MODEL_NAME=""
THINKING_BUDGET=
ENABLE_WAIT_INSERTION=""
INCLUDE_QUALITY_SCORE=""
DATASET_NAME=""
SEED=42

if [ "$THINKING_BUDGET" -eq 0 ]; then
    TEMPERATURE=0.7
    TOP_P=0.8
else
    TEMPERATURE=0.6
    TOP_P=0.95
fi

if [ "$INCLUDE_QUALITY_SCORE" = "False" ]; then
    OUTPUT_DIR="/scratch/project_462000941/members/zihao/rm4mt/post_edited_without_quality_score/${DATASET_NAME}/$(basename ${MODEL_NAME})/budget_${THINKING_BUDGET}"
else
    OUTPUT_DIR="/scratch/project_462000941/members/zihao/rm4mt/post_edited/${DATASET_NAME}/$(basename ${MODEL_NAME})/budget_${THINKING_BUDGET}"
fi

SCRIPT="post_editing_qwen.py"

echo "Post-editing files in $INPUT_DIR ..."

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

if [ "$ENABLE_WAIT_INSERTION" = "True" ]; then
    CMD="$CMD --enable_wait_insertion"
fi

if [ "$INCLUDE_QUALITY_SCORE" = "True" ]; then
    CMD="$CMD --include_quality_score"
fi

eval $CMD

echo "Done: results saved to $OUTPUT_DIR"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"