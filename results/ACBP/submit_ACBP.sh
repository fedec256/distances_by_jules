#!/bin/bash
#SBATCH --job-name=fr__ACBP
#SBATCH --output=../results/ACBP/slurm_%x_%j.out
#SBATCH --error=../results/ACBP/slurm_%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --partition=rome

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate frustra_dark

# 1. Run MCMC freezing
# Defaulting to 100,000 steps to reach local minimum
python run_mcmc_freezing.py ACBP 100000

# 2. Compute Distances (using Neff sampling strategy)
python compute_distances.py ACBP

# 3. Generate Plots (Hamming Heatmaps and Energy KDEs)
python distances_graphs_generator.py ACBP
