import numpy as np
import os
import sys
import glob
import re
import datetime
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../modules')
import alignments_functions
import mcmc_functions as mcmc

sys.path.append('.')
import plots_of_distances as plots

def analyze_basins_data(frozen_alignment, h=None, J=None):
    n_seqs, n_pos = frozen_alignment.shape

    # Find unique sequences and their counts
    unique_seqs, counts = np.unique(frozen_alignment, axis=0, return_counts=True)

    # Sort by counts in descending order (Rank)
    sort_indices = np.argsort(-counts)
    sorted_counts = counts[sort_indices]
    sorted_unique_seqs = unique_seqs[sort_indices]

    n_basins = len(sorted_counts)

    # Calculate energies if Potts model is provided
    energies = None
    if h is not None and J is not None:
        energies = np.zeros(n_basins)
        for i in range(n_basins):
            energies[i] = mcmc.E_tot(sorted_unique_seqs[i], h, J)

    return sorted_counts, sorted_unique_seqs, energies

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Tenés que pasarle el nombre de la familia. Ej: python run_thermal_basins.py ubiquitin")
        sys.exit(1)

    protein = sys.argv[1]

    data_dir = f"../data/{protein}"
    potts_path = os.path.join(data_dir, "potts.npz")
    results_base = f"../results/{protein}/thermal_ensembles/"

    if not os.path.exists(potts_path):
        print(f"Error: No se encontró potts.npz en {data_dir}")
        sys.exit(1)

    # Load Potts Model
    potts_model = np.load(potts_path)
    h = potts_model["h"]
    J = potts_model["J"]

    # Locate all simulated thermal ensembles
    ensemble_pattern = os.path.join(results_base, "simulation_of_ensemble_T_*")
    ensemble_folders = glob.glob(ensemble_pattern)

    if not ensemble_folders:
        print(f"No se encontraron simulaciones térmicas en {results_base}")
        sys.exit(0)

    print(f"Encontradas {len(ensemble_folders)} simulaciones térmicas para {protein}.")

    for sim_dir in sorted(ensemble_folders):
        print(f"\n=========================================")
        print(f"Procesando simulación térmica: {os.path.basename(sim_dir)}")

        # Extract Temperature
        match = re.search(r"T_(\d+\.\d+)", os.path.basename(sim_dir))
        T_str = match.group(1) if match else "unknown"

        # Load simulated ensemble sequences
        ensemble_seqs_path = os.path.join(sim_dir, "ensemble_of_sequences.npy")
        ensemble_energies_path = os.path.join(sim_dir, "ensemble_of_energies.npy")

        if not os.path.exists(ensemble_seqs_path):
            print(f"Warning: No se encontró ensemble_of_sequences.npy en {sim_dir}. Salteando.")
            continue

        ensemble_seqs = np.load(ensemble_seqs_path)
        n_seqs, n_pos = ensemble_seqs.shape
        print(f"Alineamiento de ensamble cargado: {n_seqs} secuencias, {n_pos} posiciones.")

        # Check if already frozen
        existing_frozen_dirs = glob.glob(os.path.join(sim_dir, "frozen_alignment_*"))
        frozen_dir = None
        for d in existing_frozen_dirs:
            f_ali_path = os.path.join(d, "frozen_alignment.npy")
            if os.path.exists(f_ali_path):
                try:
                    f_ali = np.load(f_ali_path, mmap_mode='r')
                    if f_ali.shape[0] == n_seqs:
                        frozen_dir = d
                        print(f"Ya existe una simulación completa de freezing en {frozen_dir}. Salteando freezing.")
                        break
                except Exception as e:
                    print(f"Error al leer {f_ali_path}: {e}")

        # If not already frozen, run freezing
        if frozen_dir is None:
            print(f"Iniciando freezing para {n_seqs} secuencias simulated en T={T_str} (100,000 pasos)...")
            mcmc.freezing_alignment(sim_dir, ensemble_seqs, 100000, h, J)
            # Find newly created frozen directory
            frozen_dirs = sorted(glob.glob(os.path.join(sim_dir, "frozen_alignment_*")))
            if frozen_dirs:
                frozen_dir = frozen_dirs[-1]
            else:
                print(f"Error: No se pudo localizar la carpeta de freezing generada.")
                continue

        # Paths under frozen directory
        frozen_alignment_path = os.path.join(frozen_dir, "frozen_alignment.npy")
        frozen_energies_path = os.path.join(frozen_dir, "frozen_energies.npy")
        basin_analysis_path = os.path.join(frozen_dir, "basin_analysis.npz")

        # Load frozen data
        frozen_alignment = np.load(frozen_alignment_path)
        frozen_energies = np.load(frozen_energies_path)

        # 1. Hamming Distance and Clustering analysis
        hamming_distance_path = os.path.join(frozen_dir, "hamming_distance.npy")
        hamming_orderZ_path = os.path.join(frozen_dir, "hamming_orderZ.npy")
        hamming_Z_linkage_path = os.path.join(frozen_dir, "hamming_Z_linkage_average.npy")

        if not (os.path.exists(hamming_distance_path) and os.path.exists(hamming_orderZ_path)):
            print(f"Calculando distancias Hamming y clustering jerárquico...")
            hamming_d_condensed = pdist(frozen_alignment, "hamming").astype(np.float32)
            hamming_distance_square = squareform(hamming_d_condensed).astype(np.float32)

            hamming_Z_linkage = linkage(hamming_d_condensed, method="average")
            hamming_orderZ = leaves_list(hamming_Z_linkage)

            np.save(hamming_distance_path, hamming_distance_square)
            np.save(hamming_orderZ_path, hamming_orderZ)
            np.save(hamming_Z_linkage_path, hamming_Z_linkage)
        else:
            print(f"Cargando distancias Hamming y clustering jerárquico existentes...")
            hamming_distance_square = np.load(hamming_distance_path)
            hamming_orderZ = np.load(hamming_orderZ_path)

        # 2. Basin Analysis
        if not os.path.exists(basin_analysis_path):
            print(f"Realizando análisis de cuencas...")
            counts, unique_seqs, basin_energies = analyze_basins_data(frozen_alignment, h, J)
            np.savez(basin_analysis_path, counts=counts, unique_sequences=unique_seqs, basin_energies=basin_energies)
        else:
            print(f"Cargando análisis de cuencas existente...")
            basin_data = np.load(basin_analysis_path)
            counts = basin_data["counts"]
            unique_seqs = basin_data["unique_sequences"]
            basin_energies = basin_data["basin_energies"] if "basin_energies" in basin_data else None

        # 3. Setup central plotting folder under domains
        timestamp_suffix = os.path.basename(frozen_dir).replace("frozen_alignment_", "")
        graphs_dir = f"../domains/{protein}/graphs/thermal_basins/T_{T_str}_{timestamp_suffix}/"
        os.makedirs(graphs_dir, exist_ok=True)
        print(f"Los gráficos se guardarán centralizados en: {graphs_dir}")

        # Plot 1: Hamming distance heatmap with shared energy plot below
        print(f"Generando Hamming distance heatmap...")
        heatmap_plot_path = os.path.join(graphs_dir, "hamming_distances_heatmap")
        plots.distance_and_energy_of_sequences_graph(
            hamming_distance_square,
            hamming_orderZ,
            frozen_energies,
            heatmap_plot_path
        )
        # Also copy to local frozen_dir for completeness
        plots.distance_and_energy_of_sequences_graph(
            hamming_distance_square,
            hamming_orderZ,
            frozen_energies,
            os.path.join(frozen_dir, "hamming_distances_heatmap")
        )

        # Plot 2: Basin Size vs Rank
        print(f"Generando Basin Size vs Rank plot...")
        basin_plot_path = os.path.join(graphs_dir, "basin_size_vs_rank")
        plots.plot_basin_size_vs_rank(counts, basin_plot_path)
        plots.plot_basin_size_vs_rank(counts, os.path.join(frozen_dir, "basin_size_vs_rank"))

        # Plot 3: Basin Energy vs Rank
        if basin_energies is not None:
            print(f"Generando Basin Energy vs Rank plot...")
            energy_rank_plot_path = os.path.join(graphs_dir, "basin_energy_vs_rank")
            plots.plot_energy_vs_rank(basin_energies, energy_rank_plot_path)
            plots.plot_energy_vs_rank(basin_energies, os.path.join(frozen_dir, "basin_energy_vs_rank"))

        # Plot 4: 3-panel integrated plot
        print(f"Generando gráfico de 3 paneles...")
        seq_to_count = {tuple(seq): count for seq, count in zip(unique_seqs, counts)}
        f_basin_sizes = np.array([seq_to_count[tuple(s)] for s in frozen_alignment])
        three_panel_path = os.path.join(graphs_dir, "hamming_basin_energy_3panel")
        plots.distance_basin_and_energy_graph(
            hamming_distance_square,
            hamming_orderZ,
            frozen_energies,
            f_basin_sizes,
            three_panel_path
        )
        plots.distance_basin_and_energy_graph(
            hamming_distance_square,
            hamming_orderZ,
            frozen_energies,
            f_basin_sizes,
            os.path.join(frozen_dir, "hamming_basin_energy_3panel")
        )

        # Plot 5: KDE distribution of simulated vs frozen simulated energies
        if os.path.exists(ensemble_energies_path):
            print(f"Generando KDE plot comparando energías...")
            unfrozen_energies = np.load(ensemble_energies_path)
            kde_plot_path = os.path.join(graphs_dir, "energies_KDE_distribution")

            plt.figure(figsize=(10,6))
            sns.kdeplot(x=frozen_energies, label="KDEplot Simulated Frozen", color="crimson")
            sns.kdeplot(x=unfrozen_energies, label="KDEplot Simulated Unfrozen", color="dodgerblue")
            plt.legend()
            plt.xlabel("Energy")
            plt.title(f"Energy Distribution (T = {T_str})")
            plt.savefig(f"{kde_plot_path}.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{kde_plot_path}.pdf", dpi=300, bbox_inches='tight')

            # Also save to frozen_dir
            plt.savefig(os.path.join(frozen_dir, "energies_KDE_distribution.png"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(frozen_dir, "energies_KDE_distribution.pdf"), dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Procesamiento finalizado para T={T_str}!")

    print(f"\n¡Análisis de cuencas térmicas completado para {protein}!")
