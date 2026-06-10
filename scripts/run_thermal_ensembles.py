import numpy as np
import os
import sys
import glob
from multiprocessing import cpu_count

sys.path.append('../modules')
import alignments_functions
import mcmc_functions as mcmc

def get_temperature_grid():
    """Generates the temperature grid: 0.6-0.8 (step 0.1), 0.9-1.1 (11 points), 1.2-1.8 (step 0.1)"""
    t_low = np.arange(0.6, 0.9, 0.1)
    t_transition = np.linspace(0.9, 1.1, 11)
    t_high = np.arange(1.2, 1.9, 0.1)

    # Concatenate and unique to avoid overlap at boundaries
    grid = np.unique(np.concatenate([t_low, t_transition, t_high]))
    return np.sort(grid)

def find_msa_file(protein_data_dir):
    """Finds the MSA file in the protein data directory."""
    for pattern in ["MSA.fasta", "MSA_nogap.fasta", "*.fasta"]:
        matches = glob.glob(os.path.join(protein_data_dir, pattern))
        if matches:
            return matches[0]
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Tenés que pasarle el nombre de la familia. Ej: python run_thermal_ensembles.py ubiquitin")
        sys.exit(1)

    protein = sys.argv[1]

    data_dir = f"../data/{protein}"
    potts_path = os.path.join(data_dir, "potts.npz")
    w_path = os.path.join(data_dir, "weights.npy")
    results_base = f"../results/{protein}/thermal_ensembles/"
    os.makedirs(results_base, exist_ok=True)

    if not os.path.exists(potts_path):
        print(f"Error: No se encontró potts.npz en {data_dir}")
        sys.exit(1)

    # 1. Determine Neff for Sample Size
    if os.path.exists(w_path):
        weights = np.load(w_path)
    else:
        print("Pesos no encontrados. Calculando para determinar Neff...")
        msa_path = find_msa_file(data_dir)
        if not msa_path:
            print(f"Error: No se encontró MSA en {data_dir}")
            sys.exit(1)
        seqs, _ = alignments_functions.load_msa(msa_path)
        # Efficient conversion
        MSA_np = np.zeros((len(seqs), len(seqs[0])), dtype=np.int64)
        aa_dict = alignments_functions.AA_dict_full
        for i, seq in enumerate(seqs):
            for j, char in enumerate(seq):
                MSA_np[i, j] = aa_dict.get(char, 0)
        weights = alignments_functions.compute_sequences_weight_for_many_sequences(MSA_np)
        np.save(w_path, weights)

    neff = np.sum(weights)
    n_sample = int(round(neff))
    print(f"Family: {protein} | Neff: {neff:.2f} | Target Ensemble Size: {n_sample}")

    # 2. Load Potts Model
    potts_model = np.load(potts_path)
    h = potts_model["h"]
    J = potts_model["J"]

    # 3. Iterate through Temperature Grid
    temps = get_temperature_grid()
    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))

    # Ensure n_sample is divisible by n_cores for even distribution if possible,
    # or just let generate_seq_ensemble handle it.
    # Actually, we should make sure NSeq is at least n_cores.
    if n_sample < n_cores:
        n_sample = n_cores

    for T in temps:
        print(f"\n--- Processing T = {T:.3f} ---")
        t_suffix = f"T_{T:.3f}"

        # Check if already done
        if glob.glob(os.path.join(results_base, f"simulation_of_ensemble_{t_suffix}_*")):
            print(f"Simulación para T={T:.3f} ya existe. Salteando.")
            continue

        mcmc.generate_seq_ensemble(
            path=results_base,
            num_cores=n_cores,
            Hi=h,
            Jij=J,
            NSeq=n_sample,
            temp=T,
            transient=100000, # Increased transient for thermalization
            save_each=10000,   # Steps to avoid autocorrelation
            folder_suffix=t_suffix
        )

    print("\nThermal ensembles generation completed.")
