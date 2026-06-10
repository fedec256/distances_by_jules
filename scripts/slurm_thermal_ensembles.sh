#!/bin/bash
#SBATCH --job-name=thermal__{PROTEIN}
#SBATCH --output=../results/{PROTEIN}/slurm_thermal_%j.out
#SBATCH --error=../results/{PROTEIN}/slurm_thermal_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --partition=multi

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate frustra_dark

# Run Thermal Ensembles (Multiple temperatures with high resolution transition)
python run_thermal_ensembles.py {PROTEIN}
