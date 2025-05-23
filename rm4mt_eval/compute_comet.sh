#!/bin/bash
#SBATCH --job-name=COMET-DRT-Gutenberg-Qwen3
#SBATCH --output=/scratch/project_462000941/members/zihao/slurmlog/comet/%x_%j.out
#SBATCH --error=/scratch/project_462000941/members/zihao/slurmlog/comet/%x_%j.err
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --account=project_462000675

start_time=$(date +%s)
echo "Job started at: $(date)"

# source /users/lizihao1/miniconda3/etc/profile.d/conda.sh
# conda activate /scratch/project_462000941/members/zihao/env/rm4mt_env

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/rm4mt_env/bin/activate

SCRIPT="compute_comet.py"

INPUT_ROOT="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated/DRT-Gutenberg"
OUTPUT_ROOT="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated_with_comet/DRT-Gutenberg"

# Set to true to force reprocessing of files, or false to skip already processed files
OVERWRITE=false

echo "Computing COMET scores for files in $INPUT_ROOT ..."

if [ "$OVERWRITE" = true ]; then
    echo "Mode: Overwrite existing files"
    python "$SCRIPT" \
        --input_root "$INPUT_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --overwrite
else
    echo "Mode: Skip already processed files"
    python "$SCRIPT" \
        --input_root "$INPUT_ROOT" \
        --output_root "$OUTPUT_ROOT"
fi

echo "Done: results saved to $OUTPUT_ROOT"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"