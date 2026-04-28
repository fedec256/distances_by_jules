#!/bin/bash
#SBATCH --job-name=fr__{PROTEIN}
#SBATCH --output=../results/{PROTEIN}/slurm_%x_%j.out
#SBATCH --error=../results/{PROTEIN}/slurm_%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --partition=rome

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate frustra_dark

# 1. Run MCMC freezing (if not already done)
# Using 100,000 steps as suggested in the user example
python run_mcmc_freezing.py {PROTEIN} 100000

# 2. Compute Distances (Full and/or Sampled)
python compute_distances.py {PROTEIN}

# 3. Generate Plots
python distances_graphs_generator.py {PROTEIN}
