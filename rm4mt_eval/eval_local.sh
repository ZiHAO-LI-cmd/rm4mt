#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=/scratch/project_462000941/members/zihao/slurmlog/eval_local/%x_%j.out
#SBATCH --error=/scratch/project_462000941/members/zihao/slurmlog/eval_local/%x_%j.err
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --account=project_462000675

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/rm4mt_env/bin/activate

INPUT_DIR=""
MODEL_NAME=""
THINKING_BUDGET=
SEED=42

if [ "$THINKING_BUDGET" -eq 0 ]; then
    TEMPERATURE=0.7
    TOP_P=0.8
else
    TEMPERATURE=0.6
    TOP_P=0.95
fi

DATASET_NAME=$(basename "$INPUT_DIR")
OUTPUT_DIR="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated_4_test/${DATASET_NAME}/${MODEL_NAME}/budget_${THINKING_BUDGET}"

SCRIPT="eval_local.py"

echo "Translating files in $INPUT_DIR ..."
python "$SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL_NAME" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_new_tokens 1500 \
    --thinking_budget "$THINKING_BUDGET" \
    --seed "$SEED" \
    --device_map "auto"

echo "Done: results saved to $OUTPUT_DIR"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
