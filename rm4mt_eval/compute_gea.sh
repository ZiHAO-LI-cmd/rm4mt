#!/bin/bash
#SBATCH --job-name=GEA-MULTI
#SBATCH --output=../logs/gea/%x_%j.out
#SBATCH --error=../logs/gea/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462000941

start_time=$(date +%s)
echo "Job started at: $(date)"

source ../.venv/bin/activate

SCRIPT="compute_gea.py"

# INPUT_BASE=""
# OUTPUT_BASE=""

MAX_WORKERS=5
BATCH_SIZE=100
OVERWRITE=false

DATASETS=(
  # "DRT-Gutenberg"
  # "WMT-Literary"
  # "LITEVAL-CORPUS"
)

for DATASET in "${DATASETS[@]}"; do
  INPUT_ROOT="${INPUT_BASE}/${DATASET}"
  OUTPUT_ROOT="${OUTPUT_BASE}/${DATASET}"

  echo ""
  echo "-----------------------------------------"
  echo "Processing dataset: $DATASET"
  echo "Input:  $INPUT_ROOT"
  echo "Output: $OUTPUT_ROOT"
  echo "-----------------------------------------"

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
done

echo "All datasets done."

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
