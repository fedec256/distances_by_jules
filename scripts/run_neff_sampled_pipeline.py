import numpy as np
import os
import sys
import glob
import datetime

# Add modules directory to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(repo_root, "modules"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import alignments_functions
import mcmc_functions as mcmc
from analyze_basins import analyze_basins
from compute_distances import run_hamming_analysis
from replotter import Replotter

def find_msa_file(protein_data_dir):
    """Finds the MSA file in the protein data directory."""
    for pattern in ["MSA.fasta", "MSA_nogap.fasta", "*.fasta"]:
        matches = glob.glob(os.path.join(protein_data_dir, pattern))
        if matches:
            return matches[0]
    return None

def msa_to_numpy_efficient(seqs):
    """Converts a list of Biopython sequences (or strings) to a numpy array of integers efficiently."""
    n_seqs = len(seqs)
    n_pos = len(seqs[0])
    MSA_np = np.zeros((n_seqs, n_pos), dtype=np.int64)

    aa_dict = alignments_functions.AA_dict_full
    for i, seq in enumerate(seqs):
        for j, char in enumerate(seq):
            MSA_np[i, j] = aa_dict.get(char, 0)

    return MSA_np

def run_neff_sampled_pipeline(protein, nsteps=100000):
    print(f"=== Ejecutando Neff-sampled pipeline para {protein} ===")

    domains_dir = os.path.join(repo_root, "domains", protein)
    data_dir = os.path.join(repo_root, "data", protein)
    if not os.path.exists(data_dir):
        data_dir = domains_dir

    potts_path = os.path.join(data_dir, "potts.npz")
    msa_path = find_msa_file(data_dir)

    if not msa_path:
        print(f"Error: No se encontró un archivo MSA en {data_dir}")
        sys.exit(1)

    if not os.path.exists(potts_path):
        print(f"Error: No se encontró potts.npz en {data_dir}")
        sys.exit(1)

    print(f"MSA cargado desde: {msa_path}")
    print(f"Modelo Potts desde: {potts_path}")

    # 1. Load natural MSA
    seqs, names = alignments_functions.load_msa(msa_path)
    n_seqs = len(seqs)
    MSA_np = msa_to_numpy_efficient(seqs)

    # 2. Get/Compute weights for natural MSA
    w_path = os.path.join(data_dir, "weights.npy")
    if os.path.exists(w_path):
        print(f"Cargando pesos naturales desde {w_path}")
        w_natural = np.load(w_path)
    else:
        print(f"Calculando pesos naturales para {n_seqs} secuencias...")
        w_natural = alignments_functions.compute_sequences_weight_for_many_sequences(MSA_np)
        try:
            np.save(w_path, w_natural)
            print(f"Pesos guardados en {w_path}")
        except Exception as e:
            print(f"No se pudo guardar weights.npy en {w_path}: {e}")

    # 3. Calculate Neff and sample N_eff sequences
    neff = np.sum(w_natural)
    n_sample = int(round(neff))
    if n_sample < 2:
        n_sample = min(2, n_seqs)

    print(f"Neff total: {neff:.2f} -> Seleccionando {n_sample} secuencias por muestreo ponderado...")

    prob = w_natural / w_natural.sum()
    sampled_indices = np.random.choice(n_seqs, n_sample, p=prob, replace=False)
    sampled_indices.sort()

    sampled_MSA_np = MSA_np[sampled_indices]
    sampled_seqs = [seqs[i] for i in sampled_indices]
    sampled_names = [names[i] for i in sampled_indices]
    original_weights_sampled = w_natural[sampled_indices]

    # 4. Create output directory for simulation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    simulations_base = os.path.join(repo_root, "results", protein, "simulations_of_neff_sampled_frozen_alignments")
    sim_dir = os.path.join(simulations_base, f"neff_sampled_frozen_alignment_{timestamp}")
    os.makedirs(sim_dir, exist_ok=True)
    print(f"Directorio de simulación: {sim_dir}")

    # 5. Save sampled MSA (FASTA + NPY) and original weights
    np.save(os.path.join(sim_dir, "sampled_msa.npy"), sampled_MSA_np)
    alignments_functions.write_fasta(sampled_seqs, sampled_names, os.path.join(sim_dir, "sampled_msa.fasta"))
    np.save(os.path.join(sim_dir, "original_weights.npy"), original_weights_sampled)

    # 6. Recalculate weights on the sampled MSA
    print("Recalculando pesos para las secuencias del alineamiento muestreado...")
    recalculated_weights = alignments_functions.compute_sequences_weight_for_many_sequences(sampled_MSA_np)
    np.save(os.path.join(sim_dir, "recalculated_weights.npy"), recalculated_weights)

    # 7. Perform Potts model MCMC Freezing
    potts_model = np.load(potts_path)
    h = potts_model["h"]
    J = potts_model["J"]

    # Pre-allocate array files for freezing_alignment path_in_process initialization
    np.save(os.path.join(sim_dir, "frozen_alignment.npy"), np.zeros((n_sample, sampled_MSA_np.shape[1]), dtype=np.int64))
    np.save(os.path.join(sim_dir, "frozen_energies.npy"), np.zeros(n_sample, dtype=np.float64))

    print(f"Iniciando MCMC freezing para {n_sample} secuencias muestreadas ({nsteps} pasos)...")
    mcmc.freezing_alignment(simulations_base, sampled_MSA_np, nsteps, h, J, path_in_process=sim_dir)
    print("Freezing completado.")

    # 8. Basin Analysis
    frozen_ali_path = os.path.join(sim_dir, "frozen_alignment.npy")
    print("Analizando cuencas congeladas (Basins)...")
    counts, unique_seqs, energies = analyze_basins(frozen_ali_path, h, J)
    out_basin_path = os.path.join(sim_dir, "basin_analysis.npz")
    np.savez(out_basin_path, counts=counts, unique_sequences=unique_seqs, basin_energies=energies)
    print(f"Cuencas analizadas y guardadas en {out_basin_path} (Suma de tamaños de cuencas = {np.sum(counts)})")

    # 9. Compute Hamming Distances & Linkage
    print("Calculando distancias de Hamming y dendrograma de clustering...")
    frozen_sequences_all = np.load(os.path.join(sim_dir, "frozen_alignment.npy"))
    frozen_energies_all = np.load(os.path.join(sim_dir, "frozen_energies.npy"))

    run_hamming_analysis(
        frozen_sequences_all,
        frozen_energies_all,
        recalculated_weights,
        sampled_indices,
        names,
        sim_dir,
        suffix=""
    )

    # 10. Generate Graphs
    graphs_base_dir = os.path.join(repo_root, "domains", protein, "graphs")
    replotter = Replotter(graphs_base_dir)

    # Calculate natural energies for KDE comparison
    natural_energies = None
    try:
        natural_energies = np.zeros(n_seqs)
        for i in range(n_seqs):
            natural_energies[i] = mcmc.E_tot(MSA_np[i], h, J)
    except Exception as e:
        print(f"Warning: No se pudieron calcular energías naturales: {e}")

    subfolder = os.path.join("neff_sampled_basins", f"neff_sampled_frozen_alignment_{timestamp}")
    print(f"Generando gráficos en: {os.path.join(graphs_base_dir, subfolder)}")

    replotter.plot_hamming_and_energy(sim_dir, suffix="", subfolder=subfolder)
    replotter.plot_basin_analysis(sim_dir, suffix="", subfolder=subfolder)
    replotter.plot_energy_kde(sim_dir, suffix="", subfolder=subfolder, natural_energies=natural_energies, w_natural=w_natural)

    print(f"=== Pipeline finalizado con éxito para {protein} ===")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python run_neff_sampled_pipeline.py <nombre_familia> [nsteps]")
        sys.exit(1)

    protein_arg = sys.argv[1]
    nsteps_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 100000

    run_neff_sampled_pipeline(protein_arg, nsteps_arg)
