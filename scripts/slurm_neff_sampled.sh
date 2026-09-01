#!/bin/bash
#SBATCH --job-name=neff_fzn_{PROTEIN}
#SBATCH --output=../results/{PROTEIN}/slurm_neff_sampled_%j.out
#SBATCH --error=../results/{PROTEIN}/slurm_neff_sampled_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --partition=multi

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate frustra_dark

# Run Neff-sampled freezing pipeline
python run_neff_sampled_pipeline.py {PROTEIN} 100000
