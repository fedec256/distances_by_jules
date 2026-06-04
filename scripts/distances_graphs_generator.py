import numpy as np
import os
import sys
import glob
from matplotlib import pyplot as plt, colors
import seaborn as sns
sys.path.append('../modules')
import alignments_functions
import mcmc_functions as mcmc
import plots_of_distances as plots

def generate_plots_for_type(data_path, frozen_energies, natural_energies, w_natural, w_frozen, suffix=""):
    print(f" -> Generando gráficos para {suffix or 'full'}...")
    
    out_hamming = os.path.join(data_path, f"hamming_distance{suffix}.npy")
    out_order_hamming = os.path.join(data_path, f"hamming_orderZ{suffix}.npy")

    try:
        hamming_distance = np.load(out_hamming)
        hamming_orderZ = np.load(out_order_hamming)
    except FileNotFoundError:
        print(f"  -> Archivos de distancia Hamming para {suffix} no encontrados. Salteando.")
        return

    # Hamming Heatmap
    hamming_heatmap_path = os.path.join(data_path, f"hamming_distances_heatmap{suffix}")
    plots.distance_and_energy_of_sequences_graph(hamming_distance, hamming_orderZ, frozen_energies, hamming_heatmap_path)

    # Basin Analysis Plot (only for full, suffix-less analysis usually, but we check)
    basin_file = os.path.join(data_path, "basin_analysis.npz")
    if os.path.exists(basin_file):
        print(f"  -> Graficando Basin Size vs Rank {suffix}...")
        basin_data = np.load(basin_file)
        counts = basin_data["counts"]
        basin_plot_path = os.path.join(data_path, f"basin_size_vs_rank{suffix}")
        plots.plot_basin_size_vs_rank(counts, basin_plot_path)

    # KDE
    print(f"  -> Graficando distribuciones KDE {suffix}...")
    kde_plot_path = os.path.join(data_path, f"energies_KDE_distribution{suffix}")
    plt.figure(figsize=(10,6))

    # Weights might not match sampled size, so we check
    if w_frozen is not None and len(w_frozen) == len(frozen_energies):
        sns.kdeplot(x=frozen_energies, label=f"KDEplot Frozen ({suffix or 'full'})", weights=w_frozen)
    else:
        sns.kdeplot(x=frozen_energies, label=f"KDEplot Frozen ({suffix or 'full'})")

    if w_natural is not None and len(w_natural) == len(natural_energies):
        sns.kdeplot(x=natural_energies, label="KDEplot Natural", weights=w_natural)
    else:
        sns.kdeplot(x=natural_energies, label="KDEplot Natural")

    plt.legend()
    plt.xlabel("Energy")
    plt.savefig(f"{kde_plot_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{kde_plot_path}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def find_msa_file(protein_data_dir):
    """Finds the MSA file in the protein data directory."""
    for pattern in ["MSA.fasta", "MSA_nogap.fasta", "*.fasta"]:
        matches = glob.glob(os.path.join(protein_data_dir, pattern))
        if matches:
            return matches[0]
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Tenés que pasarle el nombre de la familia. Ej: python distances_graphs_generator.py ubiquitin")
        sys.exit(1)

    protein = sys.argv[1]

    data_dir = f"../data/{protein}"
    potts_path = os.path.join(data_dir, "potts.npz")
    w_path = os.path.join(data_dir, "weights.npy")
    simulations_path = f"../results/{protein}/simulations_of_frozen_alignments/"

    msa_path = find_msa_file(data_dir)
    if not msa_path:
        print(f"Error: No se encontró un archivo MSA en {data_dir}")
        sys.exit(1)

    # Load natural data once
    print(f"Cargando datos naturales desde {msa_path}...")
    seqs, names = alignments_functions.load_msa(msa_path)

    # Efficient conversion
    n_seqs = len(seqs)
    n_pos = len(seqs[0])
    MSA_np = np.zeros((n_seqs, n_pos), dtype=np.int64)
    aa_dict = alignments_functions.AA_dict_full
    for i, seq in enumerate(seqs):
        for j, char in enumerate(seq):
            MSA_np[i, j] = aa_dict.get(char, 0)

    potts_model = np.load(potts_path)
    h = potts_model["h"]
    J = potts_model["J"]

    w_natural = np.load(w_path) if os.path.exists(w_path) else None

    print("Calculando energías naturales...")
    natural_energies = np.zeros(n_seqs)
    for i in range(n_seqs):
        natural_energies[i] = mcmc.E_tot(MSA_np[i], h, J)

    search_pattern = os.path.join(simulations_path, "frozen_alignment_*")
    simulation_folders = glob.glob(search_pattern)

    if not simulation_folders:
        print(f"No se encontraron simulaciones en {simulations_path}")
        sys.exit(0)

    for data_path in simulation_folders:
        print(f"Procesando carpeta de resultados: {data_path}")

        # Check for sampled Neff analysis (the new priority)
        path_e_neff = os.path.join(data_path, 'frozen_energies_sampled_neff.npy')
        if os.path.exists(path_e_neff):
            frozen_energies = np.load(path_e_neff)
            path_w_neff = os.path.join(data_path, 'frozen_weights_sampled_neff.npy')
            w_frozen_neff = np.load(path_w_neff) if os.path.exists(path_w_neff) else None
            generate_plots_for_type(data_path, frozen_energies, natural_energies, w_natural, w_frozen_neff, suffix="_sampled_neff")

        # Check for legacy suffix-less or "_sampled" files if they exist
        for legacy_suffix in ["", "_sampled"]:
            path_e = os.path.join(data_path, f'frozen_energies{legacy_suffix}.npy')
            if os.path.exists(path_e) and legacy_suffix != "_sampled_neff":
                frozen_energies = np.load(path_e)
                path_w = os.path.join(data_path, f'frozen_weights{legacy_suffix}.npy')
                w_frozen = np.load(path_w) if os.path.exists(path_w) else None
                generate_plots_for_type(data_path, frozen_energies, natural_energies, w_natural, w_frozen, suffix=legacy_suffix)

    print("¡Gráficos generados!")
