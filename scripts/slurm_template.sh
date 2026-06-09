#!/bin/bash
#SBATCH --job-name=fr__{PROTEIN}
#SBATCH --output=../results/{PROTEIN}/slurm_%x_%j.out
#SBATCH --error=../results/{PROTEIN}/slurm_%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --partition=multi

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate frustra_dark

# 1. Run MCMC freezing
# Defaulting to 100,000 steps to reach local minimum
python run_mcmc_freezing.py {PROTEIN} 100000

# 2. Compute Distances (using Neff sampling strategy)
python compute_distances.py {PROTEIN}

# 3. Analyze Basins (Unique sequences and counts)
python analyze_basins.py {PROTEIN}

# 4. Generate Plots (Hamming Heatmaps, Energy KDEs, and Basin Rank plots)
python distances_graphs_generator.py {PROTEIN}
