#!/bin/bash
#SBATCH --job-name=GEMBA-DRT-Gutenberg-Qwen3-14B
#SBATCH --output=/scratch/project_462000941/members/zihao/slurmlog/gemba/%x_%j.out
#SBATCH --error=/scratch/project_462000941/members/zihao/slurmlog/gemba/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462000675

start_time=$(date +%s)
echo "Job started at: $(date)"

source /users/lizihao1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/project_462000941/members/zihao/env/rm4mt_env

# module use /appl/local/csc/modulefiles/
# module load pytorch/2.5
# source /flash/project_462000941/venv/rm4mt_env/bin/activate

SCRIPT="compute_gemba.py"

INPUT_ROOT="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated_4_test/DRT-Gutenberg"
OUTPUT_ROOT="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated_with_gemba/DRT-Gutenberg"


MAX_WORKERS=5
BATCH_SIZE=100

# Set to true to force reprocessing of files, or false to skip already processed files
OVERWRITE=true

echo "Computing GEMBA scores for files in $INPUT_ROOT ..."

if [ "$OVERWRITE" = true ]; then
    echo "Mode: Overwrite existing files"
    python "$SCRIPT" \
        --input_root "$INPUT_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --max_workers "$MAX_WORKERS" \
        --batch_size "$BATCH_SIZE" \
        --overwrite
else
    echo "Mode: Skip already processed files"
    python "$SCRIPT" \
        --input_root "$INPUT_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --max_workers "$MAX_WORKERS" \
        --batch_size "$BATCH_SIZE"
fi

echo "Done: results saved to $OUTPUT_ROOT"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"