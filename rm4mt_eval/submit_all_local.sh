#!/bin/bash

DATASETS=(
  # "APE-QUEST"
  # "CAMT"
  # "DRT-Gutenberg"
  # "WMT23-Biomedical-Doc"
  # "WMT23-Biomedical-Sentence"
  # "WMT24-Biomedical"
  # "WMT-Literary"
)

MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
  "Qwen/Qwen3-32B"
)

THINKING_BUDGETS=(
  0
  100
  200
  300
  400
  500
  1000
  1500
  2000
)

TEMPLATE="eval_local.sh"

TMP_SCRIPT_DIR="./tmp_jobs"
mkdir -p "$TMP_SCRIPT_DIR"

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for budget in "${THINKING_BUDGETS[@]}"; do

      jobname="${dataset}_$(basename "$model")_b_${budget}"
      tmp_script="${TMP_SCRIPT_DIR}/${jobname}.sh"

      if [[ "$model" == *"32B"* ]]; then
        GPUS_PER_TASK=2
      else
        GPUS_PER_TASK=1
      fi

      echo "Generating job: $jobname"

      sed \
        -e "s|^#SBATCH --job-name=.*|#SBATCH --job-name=${jobname}|" \
        -e "s|^#SBATCH --gpus-per-task=.*|#SBATCH --gpus-per-task=${GPUS_PER_TASK}|" \
        -e "s|^INPUT_DIR=.*|INPUT_DIR=\"/scratch/project_462000941/members/zihao/rm4mt/rm4mt_dataset/processed/${dataset}\"|" \
        -e "s|^MODEL_NAME=.*|MODEL_NAME=\"${model}\"|" \
        -e "s|^THINKING_BUDGET=.*|THINKING_BUDGET=${budget}|" \
        "$TEMPLATE" >"$tmp_script"

      chmod +x "$tmp_script"
      sbatch "$tmp_script"

    done
  done
done
