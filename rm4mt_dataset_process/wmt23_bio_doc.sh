#!/bin/bash
#SBATCH --job-name=wmt23_bio_doc
#SBATCH --output=/scratch/project_462000675/members/zihao/slurmlog/data_process/%x_%j.out
#SBATCH --error=/scratch/project_462000675/members/zihao/slurmlog/data_process/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462000675

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5

python ./wmt23_bio_doc.py --base_path /scratch/project_462000675/members/zihao/rm4mt_dataset/WMT23-Biomedical

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
