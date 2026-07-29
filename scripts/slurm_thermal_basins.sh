#!/bin/bash
#SBATCH --job-name=th_bsn_{PROTEIN}
#SBATCH --output=../results/{PROTEIN}/slurm_thermal_basins_%j.out
#SBATCH --error=../results/{PROTEIN}/slurm_thermal_basins_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --partition=multi

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate frustra_dark

# Run thermal basin analysis
python run_thermal_basins.py {PROTEIN}
