#!/bin/bash
#SBATCH --job-name=xml2jsonl
#SBATCH --output=/scratch/project_462000675/members/zihao/slurmlog/data_process/%x_%j.out
#SBATCH --error=/scratch/project_462000675/members/zihao/slurmlog/data_process/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462000675

start_time=$(date +%s)
echo "Job started at: $(date)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module use /appl/local/csc/modulefiles/
module load pytorch/2.5

python ./wmt_literary_2_jsonl.py --threads $OMP_NUM_THREADS

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
