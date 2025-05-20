#!/bin/bash

DATASETS=(
    # "APE-QUEST"
    # "CAMT"
    "DRT-Gutenberg"
    # "WMT23-Biomedical"
    # "WMT23-Biomedical-Doc"
    # "WMT23-Biomedical-Sentence"
    # "WMT24-Biomedical"
    # "WMT-Literary"
)


MODELS=(
    "qwen3-0.6b"
    # "qwen3-1.7b"
    # "qwen3-8b"
    # "qwen3-14b"
    # "qwen3-32b"
)

THINKING_BUDGETS=(
    # 0
    500
    # 1000
    # 1500
    # 2000
)

TEMPLATE="eval.sh"

TMP_SCRIPT_DIR="./tmp_jobs"
mkdir -p "$TMP_SCRIPT_DIR"

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for budget in "${THINKING_BUDGETS[@]}"; do

      jobname="${dataset}_${model}_b_${budget}"
      tmp_script="${TMP_SCRIPT_DIR}/${jobname}.sh"

      echo "Generating job: $jobname"

      sed \
        -e "s|^#SBATCH --job-name=.*|#SBATCH --job-name=${jobname}|" \
        -e "s|^INPUT_DIR=.*|INPUT_DIR=\"/scratch/project_462000941/members/zihao/rm4mt/rm4mt_dataset/processed/${dataset}\"|" \
        -e "s|^MODEL_NAME=.*|MODEL_NAME=\"${model}\"|" \
        -e "s|^THINKING_BUDGET=.*|THINKING_BUDGET=${budget}|" \
        "$TEMPLATE" > "$tmp_script"

      chmod +x "$tmp_script"
      sbatch "$tmp_script"

    done
  done
done
