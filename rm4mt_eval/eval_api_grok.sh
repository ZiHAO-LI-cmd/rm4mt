#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=/scratch/project_462000941/members/zihao/rm4mt/logs/eval_api/%x_%j.out
#SBATCH --error=/scratch/project_462000941/members/zihao/rm4mt/logs/eval_api/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462000675

start_time=$(date +%s)
echo "Job started at: $(date)"

source /users/lizihao1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/project_462000941/members/zihao/env/rm4mt_env

INPUT_DIR=""
MODEL_NAME=""
BASE_URL="https://api.x.ai/v1"
REASONING_EFFORT="low"
SEED=42

DATASET_NAME=$(basename "$INPUT_DIR")
OUTPUT_DIR="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated/${DATASET_NAME}/${MODEL_NAME}/reasoning_effort_${REASONING_EFFORT}"

SCRIPT="eval_api_grok.py"

echo "Translating files in $INPUT_DIR ..."

CMD="python $SCRIPT \
    --input_dir \"$INPUT_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --model \"$MODEL_NAME\" \
    --base_url "$BASE_URL" \
    --seed "$SEED" \
    --reasoning_effort \"$REASONING_EFFORT\""

eval $CMD

echo "Done: results saved to $OUTPUT_DIR"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
