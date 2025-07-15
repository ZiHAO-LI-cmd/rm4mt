#!/bin/bash
#SBATCH --job-name=COMET-MULTI
#SBATCH --output=/scratch/project_462000941/members/zihao/rm4mt/logs/comet/%x_%j.out
#SBATCH --error=/scratch/project_462000941/members/zihao/rm4mt/logs/comet/%x_%j.err
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --account=project_462000675

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/rm4mt_env/bin/activate

SCRIPT="compute_comet.py"

# INPUT_BASE="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated"
# OUTPUT_BASE="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated_with_comet"

INPUT_BASE="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_wait_translated"
OUTPUT_BASE="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_wait_translated_with_comet"

OVERWRITE=false

GPU_COUNT=${SLURM_GPUS_PER_TASK}
echo "Using $GPU_COUNT GPUs based on SLURM_GPUS_PER_TASK."

DATASETS=(
  "CAMT"
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
