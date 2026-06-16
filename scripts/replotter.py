import numpy as np
import os
import glob
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure we can import plots_of_distances from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import plots_of_distances as plots

class Replotter:
    """
    A class to regenerate plots from saved simulation data.
    """
    def __init__(self, output_base_dir):
        """
        Initialize the Replotter with a base output directory.

        Args:
            output_base_dir (str): Path where the 'graphs/' folder will be created.
        """
        self.output_base_dir = output_base_dir
        os.makedirs(self.output_base_dir, exist_ok=True)

    def plot_hamming_and_energy(self, data_path, suffix="", subfolder=""):
        """Generates the Hamming distance heatmap and energy plot."""
        out_dir = os.path.join(self.output_base_dir, subfolder)
        os.makedirs(out_dir, exist_ok=True)

        try:
            hamming_distance = np.load(os.path.join(data_path, f"hamming_distance{suffix}.npy"))
            hamming_orderZ = np.load(os.path.join(data_path, f"hamming_orderZ{suffix}.npy"))
            frozen_energies = np.load(os.path.join(data_path, f"frozen_energies{suffix}.npy"))

            saving_path = os.path.join(out_dir, f"hamming_distances_heatmap{suffix}")
            plots.distance_and_energy_of_sequences_graph(hamming_distance, hamming_orderZ, frozen_energies, saving_path)
            print(f"      [OK] Hamming heatmap: {os.path.basename(saving_path)}")
        except FileNotFoundError:
            # Silently skip if files don't exist
            pass
        except Exception as e:
            print(f"      [Error] Hamming heatmap{suffix}: {e}")

    def plot_basin_analysis(self, data_path, suffix="", subfolder=""):
        """Generates Basin Size vs Rank and 3-panel plots."""
        out_dir = os.path.join(self.output_base_dir, subfolder)
        os.makedirs(out_dir, exist_ok=True)

        basin_file = os.path.join(data_path, "basin_analysis.npz")
        if not os.path.exists(basin_file):
             return

        try:
            basin_data = np.load(basin_file)
            counts = basin_data["counts"]

            # 1. Basin Size vs Rank
            plots.plot_basin_size_vs_rank(counts, os.path.join(out_dir, f"basin_size_vs_rank{suffix}"))
            print(f"      [OK] Basin Size vs Rank")

            # 2. Energy vs Rank
            if "basin_energies" in basin_data:
                plots.plot_energy_vs_rank(basin_data["basin_energies"], os.path.join(out_dir, f"basin_energy_vs_rank{suffix}"))
                print(f"      [OK] Basin Energy vs Rank")

            # 3. 3-panel Distance Matrix (Basin Size + Heatmap + Energy)
            frozen_alignment_path = os.path.join(data_path, f"frozen_alignment{suffix}.npy")
            hamming_distance_path = os.path.join(data_path, f"hamming_distance{suffix}.npy")
            hamming_orderZ_path = os.path.join(data_path, f"hamming_orderZ{suffix}.npy")
            frozen_energies_path = os.path.join(data_path, f"frozen_energies{suffix}.npy")

            if all(os.path.exists(p) for p in [frozen_alignment_path, hamming_distance_path, hamming_orderZ_path, frozen_energies_path]):
                f_ali = np.load(frozen_alignment_path)
                u_seqs = basin_data["unique_sequences"]
                u_counts = basin_data["counts"]

                # Map each sequence to its count
                seq_to_count = {tuple(seq): count for seq, count in zip(u_seqs, u_counts)}
                f_basin_sizes = np.array([seq_to_count[tuple(s)] for s in f_ali])

                hamming_distance = np.load(hamming_distance_path)
                hamming_orderZ = np.load(hamming_orderZ_path)
                frozen_energies = np.load(frozen_energies_path)

                saving_path = os.path.join(out_dir, f"hamming_basin_energy_3panel{suffix}")
                plots.distance_basin_and_energy_graph(hamming_distance, hamming_orderZ, frozen_energies, f_basin_sizes, saving_path)
                print(f"      [OK] 3-panel plot: {os.path.basename(saving_path)}")
        except Exception as e:
            print(f"      [Error] Basin analysis{suffix}: {e}")

    def plot_energy_kde(self, data_path, suffix="", subfolder="", natural_energies=None, w_natural=None):
        """Generates KDE plot of energy distributions."""
        out_dir = os.path.join(self.output_base_dir, subfolder)
        os.makedirs(out_dir, exist_ok=True)

        try:
            frozen_energies = np.load(os.path.join(data_path, f"frozen_energies{suffix}.npy"))
            w_frozen_path = os.path.join(data_path, f"frozen_weights{suffix}.npy")
            w_frozen = np.load(w_frozen_path) if os.path.exists(w_frozen_path) else None

            kde_plot_path = os.path.join(out_dir, f"energies_KDE_distribution{suffix}")
            plt.figure(figsize=(10,6))

            if w_frozen is not None and len(w_frozen) == len(frozen_energies):
                sns.kdeplot(x=frozen_energies, label=f"KDEplot Frozen ({suffix or 'full'})", weights=w_frozen)
            else:
                sns.kdeplot(x=frozen_energies, label=f"KDEplot Frozen ({suffix or 'full'})")

            if natural_energies is not None:
                if w_natural is not None and len(w_natural) == len(natural_energies):
                    sns.kdeplot(x=natural_energies, label="KDEplot Natural", weights=w_natural)
                else:
                    sns.kdeplot(x=natural_energies, label="KDEplot Natural")

            plt.legend()
            plt.xlabel("Energy")
            plt.savefig(f"{kde_plot_path}.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{kde_plot_path}.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      [OK] Energy KDE: {os.path.basename(kde_plot_path)}")
        except Exception as e:
            print(f"      [Error] Energy KDE{suffix}: {e}")

    def plot_thermal(self, thermal_results_dir, subfolder="thermal"):
        """Generates thermal properties and energy distribution plots."""
        out_dir = os.path.join(self.output_base_dir, subfolder)
        os.makedirs(out_dir, exist_ok=True)

        # 1. Thermal Properties
        summary_path = os.path.join(thermal_results_dir, "thermal_analysis_summary.npz")
        if os.path.exists(summary_path):
            try:
                summary = np.load(summary_path)
                plots.plot_thermal_properties(
                    summary["temperatures"],
                    summary["mean_energies"],
                    summary["cv_values"],
                    os.path.join(out_dir, "thermal_properties_vs_T")
                )
                print(f"      [OK] Thermal properties vs T")
            except Exception as e:
                print(f"      [Error] Thermal properties: {e}")

        # 2. Distributions vs T (KDE)
        sim_dirs = glob.glob(os.path.join(thermal_results_dir, "simulation_of_ensemble_T_*"))
        data_by_temp = {}
        for sim_dir in sim_dirs:
            match = re.search(r"T_(\d+\.\d+)", os.path.basename(sim_dir))
            if match:
                T = float(match.group(1))
                energy_path = os.path.join(sim_dir, "ensemble_of_energies.npy")
                if os.path.exists(energy_path):
                    try:
                        data_by_temp[T] = np.load(energy_path)
                    except Exception:
                        pass

        if data_by_temp:
            try:
                plots.plot_thermal_energy_distributions(data_by_temp, os.path.join(out_dir, "energy_distributions_vs_T"))
                print(f"      [OK] Energy distributions vs T")
            except Exception as e:
                print(f"      [Error] Thermal distributions: {e}")
