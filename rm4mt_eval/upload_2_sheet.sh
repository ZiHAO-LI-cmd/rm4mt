#!/bin/bash

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/rm4mt_env/bin/activate

# Define arrays for configurations
declare -a DESCRIPTIONS=(
    "Comet Score"
    "Comet Score (Wait)"
    "Gemini GRB&GRF Score"
    "Gemini GRB&GRF Score (Wait)"
    "Gemini GEA Score"
    "Gemini GEA Score (Wait)"
    "Thinking Token Length"
    "Thinking Token Length (Wait)"
)

declare -a ROOT_DIRECTORIES=(
    "/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated_with_comet"
    "/scratch/project_462000941/members/zihao/rm4mt/rm4mt_wait_translated_with_comet"
    "/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated_with_grb_grf"
    "/scratch/project_462000941/members/zihao/rm4mt/rm4mt_wait_translated_with_grb_grf"
    "/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated_with_gea"
    "/scratch/project_462000941/members/zihao/rm4mt/rm4mt_wait_translated_with_gea"
    "/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated"
    "/scratch/project_462000941/members/zihao/rm4mt/rm4mt_wait_translated"
)

declare -a GOOGLE_SHEET_URLS=(
    "https://docs.google.com/spreadsheets/d/1yhLFAwm-sFAG2cDxDBQ6C0smcM_f5uQxYuy_clvgueM/edit?usp=sharing"
    "https://docs.google.com/spreadsheets/d/1LHsMMvKhTEgagmZKQl7ax1H4BJyDBImYMbeE8sNPjJw/edit?usp=sharing"
    "https://docs.google.com/spreadsheets/d/19sowSDpBO42OSXFeBM92UBp1H2MZkQQWzEAPMpZz9q4/edit?usp=sharing"
    "https://docs.google.com/spreadsheets/d/1PRIfwNA4l2TG3oMFlgM8LIGuxvCH8prfgEvbGMDhqIY/edit?usp=sharing"
    "https://docs.google.com/spreadsheets/d/16nVSzWScuuYThX2ZgQAyZ0V5ip5d1GFPKCABjG6IHf8/edit?usp=sharing"
    "https://docs.google.com/spreadsheets/d/11WAU3v3uKN1wV3G7crmGDyo3u4QX4aGlhWpQN7tn9As/edit?usp=sharing"
    "https://docs.google.com/spreadsheets/d/1k_zR5v7em_iz1J3K6p4qRYvFrzbFDOjI8lLWtPK3YCM/edit?usp=sharing"
    "https://docs.google.com/spreadsheets/d/1pSBxhdqGvu2I5z_8C6l6pePV-IkIl31MM83Py_ILRqU/edit?usp=sharing"
)

CREDENTIALS_JSON="./rm4mt-463314-3ce1280ee29c.json"
PYTHON_SCRIPT="./upload_2_sheet.py"

# Loop through all configurations and run the Python script
echo "Starting data upload to Google Sheets for all configurations..."

for i in "${!DESCRIPTIONS[@]}"; do
    DESCRIPTION="${DESCRIPTIONS[$i]}"
    ROOT_DIRECTORY="${ROOT_DIRECTORIES[$i]}"
    GOOGLE_SHEET_URL="${GOOGLE_SHEET_URLS[$i]}"
    
    echo "----------------------------------------"
    echo "Processing configuration $((i+1))/${#DESCRIPTIONS[@]}: $DESCRIPTION"
    echo "Root directory: $ROOT_DIRECTORY"
    echo "Google Sheet URL: $GOOGLE_SHEET_URL"
    echo "----------------------------------------"
    
    python "$PYTHON_SCRIPT" \
        --root_dir "$ROOT_DIRECTORY" \
        --credentials_file "$CREDENTIALS_JSON" \
        --sheet_url "$GOOGLE_SHEET_URL"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $DESCRIPTION"
    else
        echo "✗ Failed to process: $DESCRIPTION"
    fi
    echo ""
done

echo "All configurations processed. Script finished."