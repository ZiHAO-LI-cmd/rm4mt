#!/bin/bash
#SBATCH --job-name=COMET-MULTI
#SBATCH --output=../logs/comet/%x_%j.out
#SBATCH --error=../logs/comet/%x_%j.err
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --account=project_462000941

start_time=$(date +%s)
echo "Job started at: $(date)"

source ../.venv/bin/activate

SCRIPT="compute_comet.py"

# INPUT_BASE=""
# OUTPUT_BASE=""

OVERWRITE=false

GPU_COUNT=${SLURM_GPUS_PER_TASK}
echo "Using $GPU_COUNT GPUs based on SLURM_GPUS_PER_TASK."

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
          --gpu_num "$GPU_COUNT" \
          --overwrite
  else
      echo "Mode: Skip already processed files"
      python "$SCRIPT" \
          --input_root "$INPUT_ROOT" \
          --output_root "$OUTPUT_ROOT" \
          --gpu_num "$GPU_COUNT"
  fi

  echo "Completed dataset: $DATASET"
done

echo "All datasets done."

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
