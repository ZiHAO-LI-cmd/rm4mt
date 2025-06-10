#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=/scratch/project_2008161/members/zihao/slurm_log/eval_gemini/%x_%j.out
#SBATCH --error=/scratch/project_2008161/members/zihao/slurm_log/eval_gemini/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --account=project_2008161

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /scratch/project_2008161/members/zihao/venv/rm4mt_env/bin/activate
export HF_HOME="/scratch/project_2008161/cache"

INPUT_DIR=""
MODEL_NAME=""
THINKING_BUDGET=
SEED=42

DATASET_NAME=$(basename "$INPUT_DIR")
OUTPUT_DIR="/scratch/project_2008161/members/zihao/rm4mt/rm4mt_translated/${DATASET_NAME}/${MODEL_NAME}/budget_${THINKING_BUDGET}"

SCRIPT="eval_api_gemini.py"

echo "Translating files in $INPUT_DIR ..."
python "$SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL_NAME" \
    --seed "$SEED" \
    --thinking_budget "$THINKING_BUDGET"

echo "Done: results saved to $OUTPUT_DIR"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
