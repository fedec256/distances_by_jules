import os
import glob
import numpy as np
import sys
from replotter import Replotter

# Add modules to path for alignments_functions and mcmc_functions
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(repo_root, "modules"))
import alignments_functions
import mcmc_functions as mcmc

def find_msa_file(protein_data_dir):
    """Finds the MSA file in the protein data directory."""
    for pattern in ["MSA.fasta", "MSA_nogap.fasta", "*.fasta"]:
        matches = glob.glob(os.path.join(protein_data_dir, pattern))
        if matches:
            return matches[0]
    return None

def main():
    # Base directories relative to repo root
    domains_dir = os.path.join(repo_root, "domains")
    results_dir_base = os.path.join(repo_root, "results")

    if not os.path.exists(domains_dir):
        print(f"Error: Domains directory {domains_dir} not found.")
        return

    # Get all protein families
    families = sorted([d for d in os.listdir(domains_dir) if os.path.isdir(os.path.join(domains_dir, d))])
    print(f"Encontradas {len(families)} familias en {domains_dir}")

    for protein in families:
        print(f"\n>>> Procesando familia: {protein}")

        data_dir = os.path.join(repo_root, "data", protein)
        if not os.path.exists(data_dir):
            data_dir = os.path.join(domains_dir, protein)

        potts_path = os.path.join(data_dir, "potts.npz")
        w_path = os.path.join(data_dir, "weights.npy")
        msa_path = find_msa_file(data_dir)

        natural_energies = None
        w_natural = np.load(w_path) if os.path.exists(w_path) else None

        if os.path.exists(potts_path) and msa_path:
            print(f"    - Cargando datos naturales para KDE...")
            try:
                seqs, _ = alignments_functions.load_msa(msa_path)
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

                natural_energies = np.zeros(n_seqs)
                for i in range(n_seqs):
                    natural_energies[i] = mcmc.E_tot(MSA_np[i], h, J)
            except Exception as e:
                print(f"    - [Error] No se pudieron calcular energías naturales: {e}")

        # Target directory for graphs
        graphs_base_dir = os.path.join(domains_dir, protein, "graphs")
        os.makedirs(graphs_base_dir, exist_ok=True)

        replotter = Replotter(graphs_base_dir)

        # 1. Process Frozen Alignments (Basins and Distances)
        frozen_results_base = os.path.join(results_dir_base, protein, "simulations_of_frozen_alignments")
        if os.path.exists(frozen_results_base):
            sim_dirs = sorted(glob.glob(os.path.join(frozen_results_base, "frozen_alignment_*")))

            for sim_dir in sim_dirs:
                timestamp = os.path.basename(sim_dir).replace("frozen_alignment_", "")
                print(f"    - Simulacion: {timestamp}")

                # Check for regular files and sampled_neff files
                suffixes = ["", "_sampled_neff", "_sampled"]
                for suffix in suffixes:
                    # Plot Hamming Heatmap
                    replotter.plot_hamming_and_energy(sim_dir, suffix=suffix, subfolder=timestamp)
                    # Plot Basin Analysis
                    replotter.plot_basin_analysis(sim_dir, suffix=suffix, subfolder=timestamp)
                    # Plot KDE
                    replotter.plot_energy_kde(sim_dir, suffix=suffix, subfolder=timestamp,
                                             natural_energies=natural_energies, w_natural=w_natural)
        else:
            print(f"    - No se encontraron resultados de simulaciones congeladas.")

        # 2. Process Thermal Ensembles
        thermal_results_base = os.path.join(results_dir_base, protein, "thermal_ensembles")
        if os.path.exists(thermal_results_base):
            print(f"    - Generando gráficos térmicos...")
            replotter.plot_thermal(thermal_results_base, subfolder="thermal")
        else:
            print(f"    - No se encontraron resultados térmicos.")

    print("\n¡Proceso finalizado! Los gráficos están en las carpetas 'graphs/' de cada dominio.")

if __name__ == "__main__":
    main()
