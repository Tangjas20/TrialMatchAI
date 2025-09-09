#!/bin/bash
#SBATCH --job-name=generate_cluster_patients   # Job name
#SBATCH --output=logs/cluster_%j.out   # Output file
#SBATCH --error=logs/cluster_%j.err    # Error file
#SBATCH --partition=gpu-h100,gpu-l40             # Partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --gres=gpu:1                     # Request one GPU
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem=48G                        # Memory per node

module load nvidia/cuda/12.8
echo "CUDA loaded"
module load anaconda
echo "Anaconda loaded"

conda activate unsloth_env
echo "Conda environment activated"

cd /scratch/mgeorges/TrialMatchAI/TrialMatchAI/source || { echo "Directory not found"; exit 1; }
DATA_PATH=../data

python -m Matcher.generate_cluster_patients \
    --clusters_file $DATA_PATH/clusters_metadata.csv \
    --trials_file $DATA_PATH/cluster_trials \
    --output_file $DATA_PATH/cluster_100_patients_dataset.json \
    --patients_per_cluster 100