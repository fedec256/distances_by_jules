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

def generate_plots_for_type(data_path, frozen_energies, natural_energies, w_natural, w, suffix=""):
    print(f" -> Generando gráficos para {suffix or 'full'}...")
    
    out_hi = os.path.join(data_path, f"hi_distance{suffix}.npy")
    out_order_hi = os.path.join(data_path, f"hi_orderZ{suffix}.npy")
    out_hamming = os.path.join(data_path, f"hamming_distance{suffix}.npy")
    out_order_hamming = os.path.join(data_path, f"hamming_orderZ{suffix}.npy")

    try:
        hi_distance = np.load(out_hi)
        hi_orderZ = np.load(out_order_hi)
        hamming_distance = np.load(out_hamming)
        hamming_orderZ = np.load(out_order_hamming)
    except FileNotFoundError:
        print(f"  -> Archivos de distancia para {suffix} no encontrados. Salteando.")
        return

    # Heatmaps
    hi_heatmap_path = os.path.join(data_path, f"hi_distances_heatmap{suffix}")
    plots.distance_and_energy_of_sequences_graph(hi_distance, hi_orderZ, frozen_energies, hi_heatmap_path)

    hamming_heatmap_path = os.path.join(data_path, f"hamming_distances_heatmap{suffix}")
    plots.distance_and_energy_of_sequences_graph(hamming_distance, hamming_orderZ, frozen_energies, hamming_heatmap_path)

    # KDE
    print(f"  -> Graficando distribuciones KDE {suffix}...")
    kde_plot_path = os.path.join(data_path, f"energies_KDE_distribution{suffix}")
    plt.figure(figsize=(10,6))

    # Weights might not match sampled size, so we check
    if w is not None and len(w) == len(frozen_energies):
        sns.kdeplot(x=frozen_energies, label=f"KDEplot Frozen ({suffix or 'full'})", weights=w)
    else:
        sns.kdeplot(x=frozen_energies, label=f"KDEplot Frozen ({suffix or 'full'})")

    sns.kdeplot(x=natural_energies, label="KDEplot Natural", weights=w_natural)

    plt.legend()
    plt.xlabel("Energy")
    plt.savefig(f"{kde_plot_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{kde_plot_path}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Tenés que pasarle el nombre de la familia. Ej: python distances_graphs_generator.py ubiquitin")
        sys.exit(1)

    protein = sys.argv[1]

    data_dir = f"../data/{protein}"
    msa_path = os.path.join(data_dir, "MSA.fasta")
    potts_path = os.path.join(data_dir, "potts.npz")
    w_path = os.path.join(data_dir, "weights.npy")
    simulations_path = f"../results/{protein}/simulations_of_frozen_alignments/"

    search_pattern = os.path.join(simulations_path, "frozen_alignment_*")
    simulation_folders = glob.glob(search_pattern)

    if not simulation_folders:
        print(f"No se encontraron simulaciones en {simulations_path}")
        sys.exit(0)

    # Load natural data once
    print("Cargando datos naturales...")
    seqs, names = alignments_functions.load_msa(msa_path)
    MSA_np = alignments_functions.MSA_to_numpy(np.array([[c for c in str(s)] for s in seqs]))

    potts_model = np.load(potts_path)
    h = potts_model["h"]
    J = potts_model["J"]

    w_natural = np.load(w_path) if os.path.exists(w_path) else None

    natural_energies = []
    for i in range(len(seqs)):
        natural_energies.append(mcmc.E_tot(MSA_np[i], h, J))
    natural_energies = np.array(natural_energies)

    for data_path in simulation_folders:
        print(f"Procesando: {data_path}")

        # Check for full analysis
        path_e_full = os.path.join(data_path, 'frozen_energies.npy')
        if os.path.exists(path_e_full):
            frozen_energies = np.load(path_e_full)
            generate_plots_for_type(data_path, frozen_energies, natural_energies, w_natural, w_natural, suffix="")

        # Check for sampled analysis
        path_e_sampled = os.path.join(data_path, 'frozen_energies_sampled.npy')
        if os.path.exists(path_e_sampled):
            sampled_energies = np.load(path_e_sampled)
            # For sampled, we don't have individual weights usually, so pass None or a subset if possible
            generate_plots_for_type(data_path, sampled_energies, natural_energies, w_natural, None, suffix="_sampled")

    print("Ya miramos y graficamos en todas las carpetas!")
