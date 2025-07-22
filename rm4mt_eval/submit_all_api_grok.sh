#!/bin/bash

DATASETS=(
  # "CAMT"
  "DRT-Gutenberg"
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


MODELS=(
  "grok-3-mini"
)

REASONING_EFFORTS=(
  # "low"
  "high"
)

TEMPLATE="eval_api_grok.sh"

TMP_SCRIPT_DIR="./tmp_jobs"
mkdir -p "$TMP_SCRIPT_DIR"

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for reasoning_effort in "${REASONING_EFFORTS[@]}"; do

      jobname="${dataset}_${model}_re_${reasoning_effort}"
      tmp_script="${TMP_SCRIPT_DIR}/${jobname}.sh"

      echo "Generating job: $jobname"

      sed \
        -e "s|^#SBATCH --job-name=.*|#SBATCH --job-name=${jobname}|" \
        -e "s|^INPUT_DIR=.*|INPUT_DIR=\"/scratch/project_462000941/members/zihao/rm4mt/rm4mt_dataset/processed/${dataset}\"|" \
        -e "s|^MODEL_NAME=.*|MODEL_NAME=\"${model}\"|" \
        -e "s|^REASONING_EFFORT=.*|REASONING_EFFORT=\"${reasoning_effort}\"|" \
        "$TEMPLATE" > "$tmp_script"

      chmod +x "$tmp_script"
      sbatch "$tmp_script"

    done
  done
done
