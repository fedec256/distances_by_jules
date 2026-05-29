import numpy as np
import os
import sys
import glob

sys.path.append('../modules')
import alignments_functions
import mcmc_functions as mcmc

def find_msa_file(protein_data_dir):
    """Finds the MSA file in the protein data directory."""
    # Priority 1: MSA.fasta
    path = os.path.join(protein_data_dir, "MSA.fasta")
    if os.path.exists(path):
        return path

    # Priority 2: MSA_nogap.fasta
    path = os.path.join(protein_data_dir, "MSA_nogap.fasta")
    if os.path.exists(path):
        return path

    # Priority 3: Any .fasta file
    fasta_files = glob.glob(os.path.join(protein_data_dir, "*.fasta"))
    if fasta_files:
        return fasta_files[0]

    return None

def msa_to_numpy_efficient(seqs):
    """Converts a list of Biopython sequences (or strings) to a numpy array of integers efficiently."""
    n_seqs = len(seqs)
    n_pos = len(seqs[0])
    MSA_np = np.zeros((n_seqs, n_pos), dtype=np.int64)

    # Using the dictionary from alignments_functions
    aa_dict = alignments_functions.AA_dict_full

    for i, seq in enumerate(seqs):
        for j, char in enumerate(seq):
            MSA_np[i, j] = aa_dict.get(char, 0)

    return MSA_np

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: Tenés que pasarle el nombre de la familia y nsteps. Ej: python run_mcmc_freezing.py ubiquitin 100000")
        sys.exit(1)

    protein = sys.argv[1]
    nsteps = int(sys.argv[2])

    data_dir = f"../data/{protein}"
    potts_path = os.path.join(data_dir, "potts.npz")
    simulations_path = f"../results/{protein}/simulations_of_frozen_alignments/"

    msa_path = find_msa_file(data_dir)
    if not msa_path:
        print(f"Error: No se encontró un archivo MSA en {data_dir}")
        sys.exit(1)

    print(f"Usando MSA: {msa_path}")

    if not os.path.exists(potts_path):
        print(f"Error: No se encontró potts.npz en {data_dir}")
        sys.exit(1)

    # 1. Load MSA
    seqs, names = alignments_functions.load_msa(msa_path)
    n_seqs = len(seqs)

    # 2. Check if already computed
    # We look for any folder in simulations_path that contains a complete frozen_alignment.npy
    # Note: mcmc.freezing_alignment creates subfolders like frozen_alignment_TIMESTAMP/
    existing_sims = glob.glob(os.path.join(simulations_path, "frozen_alignment_*"))
    for sim_dir in existing_sims:
        frozen_ali_path = os.path.join(sim_dir, "frozen_alignment.npy")
        if os.path.exists(frozen_ali_path):
            try:
                frozen_ali = np.load(frozen_ali_path, mmap_mode='r') # Use mmap to avoid loading huge files just for shape check
                if frozen_ali.shape[0] == n_seqs:
                    print(f"Ya existe una simulación completa en {sim_dir}. Salteando freezing.")
                    sys.exit(0)
            except Exception as e:
                print(f"Error al leer {frozen_ali_path}: {e}")

    # 3. Prepare data for MCMC
    potts_model = np.load(potts_path)
    h = potts_model["h"]
    J = potts_model["J"]

    print(f"Convirtiendo MSA ({n_seqs} secuencias) a numpy...")
    MSA_np = msa_to_numpy_efficient(seqs)

    # 4. Run freezing
    print(f"Iniciando freezing para {protein} ({nsteps} pasos)...")
    mcmc.freezing_alignment(simulations_path, MSA_np, nsteps, h, J)
    print("Freezing completado.")
