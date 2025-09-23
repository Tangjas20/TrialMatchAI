#!/bin/bash
#SBATCH --job-name=ablation_study   # Job name
#SBATCH --output=logs/ablation_%j.out   # Output file
#SBATCH --error=logs/ablation_%j.err    # Error file
#SBATCH --partition=gpu-h100             # Partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --gres=gpu:1                     # Request one GPU
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem=96G                        # Memory per node

module load nvidia/cuda/12.8
echo "CUDA loaded"
module load anaconda
echo "Anaconda loaded"

conda activate trialmatchai
echo "Conda environment activated"
echo nvcc --version
nvcc --version

# Ensure flash-attn is installed on node
echo "Checking and installing flash-attn if necessary..."
pip show flash-attn &> /dev/null
if [ $? -ne 0 ]; then
    echo "flash-attn not found. Installing..."
    pip install flash-attn --no-build-isolation
else
    echo "flash-attn is already installed."
fi

# Check if TREC year argument is provided, otherwise default to 2021
if $1; then
    TREC_YEAR=$1
else
    echo "No TREC year provided. Defaulting to 2021."
    TREC_YEAR=2021
fi
PATIENT_IDS=P001
echo "TREC Year: $TREC_YEAR"
echo "Starting ablation study for TREC year $TREC_YEAR"

# Set up SSH tunnel to forward local port 9200 to remote Elasticsearch server, if not already set up
pgrep -f "ssh -4 -f -N -L 9200:localhost:9200 cali3" > /dev/null
if [ $? -ne 0 ]; then
    echo "Setting up SSH tunnel to Elasticsearch server..."
    ssh -4 -f -N -L 9200:localhost:9200 cali3
else
    echo "SSH tunnel already established."
fi

cd /scratch/mgeorges/TrialMatchAI/TrialMatchAI/source || { echo "Directory not found"; exit 1; }

python -m Matcher.ablation_study \
     --ablation-config /home/testgpu/TrialMatchAI/source/Matcher/config/config.json \
     --trec-ground-truth /home/testgpu/TrialGPT/dataset/trec_$TREC_YEAR/qrels/test.tsv \
     --patients-file ../data/processed_patients${TREC_YEAR:2:2}.json \
     --output-dir ../results/TREC${TREC_YEAR:2:2} \
     --patient $PATIENT_IDS