#!/bin/bash
#SBATCH --job-name=GRB&GRF-MULTI
#SBATCH --output=../logs/grb_grf/%x_%j.out
#SBATCH --error=../logs/grb_grf/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462000941

start_time=$(date +%s)
echo "Job started at: $(date)"

source ../.venv/bin/activate

SCRIPT="compute_grb_grf.py"

# INPUT_BASE=""
# OUTPUT_BASE=""

MAX_WORKERS=5
BATCH_SIZE=100
OVERWRITE=false

DATASETS=(
  # "CAMT"
  # "DRT-Gutenberg"
  # "WMT23-Biomedical-Doc"
  # "WMT23-Biomedical-Sentence"
  # "WMT24-Biomedical"
  # "WMT-Literary"
  # "LITEVAL-CORPUS"
  # "CommonsenseMT-Contextless"
  # "CommonsenseMT-Contextual"
  # "CommonsenseMT-Lexical"
  # "RTT"
  # "RAGtrans"
  # "RAGtrans_without_doc"
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
