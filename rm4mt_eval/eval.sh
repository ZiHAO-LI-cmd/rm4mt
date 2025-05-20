#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=/scratch/project_462000941/members/zihao/slurmlog/eval/%x_%j.out
#SBATCH --error=/scratch/project_462000941/members/zihao/slurmlog/eval/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462000675

start_time=$(date +%s)
echo "Job started at: $(date)"

source /users/lizihao1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/project_462000941/members/zihao/env/rm4mt_env

INPUT_DIR=""
MODEL_NAME=""
BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
THINKING_BUDGET=
SEED=42
TOP_K=20

if [ "$THINKING_BUDGET" -eq 0 ]; then
    ENABLE_THINKING=""
    TEMPERATURE=0.7
    TOP_P=0.8
else
    ENABLE_THINKING="--enable_thinking"
    TEMPERATURE=0.6
    TOP_P=0.95
fi

DATASET_NAME=$(basename "$INPUT_DIR")
OUTPUT_DIR="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated/${DATASET_NAME}/${MODEL_NAME}/budget_${THINKING_BUDGET}"

SCRIPT="eval.py"

echo "Translating files in $INPUT_DIR ..."
python "$SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL_NAME" \
    --base_url "$BASE_URL" \
    --seed "$SEED" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --thinking_budget "$THINKING_BUDGET" \
    $ENABLE_THINKING

echo "Done: results saved to $OUTPUT_DIR"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
