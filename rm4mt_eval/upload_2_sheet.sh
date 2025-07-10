#!/bin/bash

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/rm4mt_env/bin/activate

# Comet Score
# ROOT_DIRECTORY="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated_with_comet"
# GOOGLE_SHEET_URL="https://docs.google.com/spreadsheets/d/1yhLFAwm-sFAG2cDxDBQ6C0smcM_f5uQxYuy_clvgueM/edit?usp=sharing"

# Comet Score (Wait)
# ROOT_DIRECTORY="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_wait_translated_with_comet"
# GOOGLE_SHEET_URL="https://docs.google.com/spreadsheets/d/1LHsMMvKhTEgagmZKQl7ax1H4BJyDBImYMbeE8sNPjJw/edit?usp=sharing"

# Gemini GRB&GRF Score
# ROOT_DIRECTORY="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated_with_grb_grf"
# GOOGLE_SHEET_URL="https://docs.google.com/spreadsheets/d/19sowSDpBO42OSXFeBM92UBp1H2MZkQQWzEAPMpZz9q4/edit?usp=sharing"

# Gemini GRB&GRF Score (Wait)
# ROOT_DIRECTORY="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_wait_translated_with_grb_grf"
# GOOGLE_SHEET_URL="https://docs.google.com/spreadsheets/d/1PRIfwNA4l2TG3oMFlgM8LIGuxvCH8prfgEvbGMDhqIY/edit?usp=sharing"

# Thingking Token Length
# ROOT_DIRECTORY="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_translated"
# GOOGLE_SHEET_URL="https://docs.google.com/spreadsheets/d/1k_zR5v7em_iz1J3K6p4qRYvFrzbFDOjI8lLWtPK3YCM/edit?usp=sharing"

# Thingking Token Length (Wait)
# ROOT_DIRECTORY="/scratch/project_462000941/members/zihao/rm4mt/rm4mt_wait_translated"
# GOOGLE_SHEET_URL="https://docs.google.com/spreadsheets/d/1pSBxhdqGvu2I5z_8C6l6pePV-IkIl31MM83Py_ILRqU/edit?usp=sharing"

CREDENTIALS_JSON="./rm4mt-463314-3ce1280ee29c.json"
PYTHON_SCRIPT="./upload_2_sheet.py"

# Run the Python script with the defined arguments
echo "Starting data upload to Google Sheets..."
python "$PYTHON_SCRIPT" \
    --root_dir "$ROOT_DIRECTORY" \
    --credentials_file "$CREDENTIALS_JSON" \
    --sheet_url "$GOOGLE_SHEET_URL"

echo "Script finished."