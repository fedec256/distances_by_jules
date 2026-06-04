#!/bin/bash
#SBATCH --job-name=basin__{PROTEIN}
#SBATCH --output=../results/{PROTEIN}/slurm_basin_%j.out
#SBATCH --error=../results/{PROTEIN}/slurm_basin_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --partition=rome

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate frustra_dark

# 1. Analyze Basins (Unique sequences and counts)
python analyze_basins.py {PROTEIN}

# 2. Re-generate Plots (to include Basin Size vs Rank)
python distances_graphs_generator.py {PROTEIN}
