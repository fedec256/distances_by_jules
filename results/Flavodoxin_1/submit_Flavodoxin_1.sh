#!/bin/bash
#SBATCH --job-name=fr__Flavodoxin_1
#SBATCH --output=../results/Flavodoxin_1/slurm_%x_%j.out
#SBATCH --error=../results/Flavodoxin_1/slurm_%x_%j.err
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
python run_mcmc_freezing.py Flavodoxin_1 100000

# 2. Compute Distances (using Neff sampling strategy)
python compute_distances.py Flavodoxin_1

# 3. Generate Plots (Hamming Heatmaps and Energy KDEs)
python distances_graphs_generator.py Flavodoxin_1
