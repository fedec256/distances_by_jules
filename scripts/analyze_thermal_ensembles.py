import numpy as np
import os
import sys
import glob
import re

sys.path.append('../modules')
import plots_of_distances as plots

def analyze_protein_thermal(protein):
    results_base = f"../results/{protein}/thermal_ensembles/"

    # Find all simulation folders and extract temperature from folder name
    # Folder pattern: simulation_of_ensemble_T_X.XXX_TIMESTAMP
    sim_dirs = glob.glob(os.path.join(results_base, "simulation_of_ensemble_T_*"))

    if not sim_dirs:
        print(f"No se encontraron simulaciones térmicas para {protein}.")
        return

    data_by_temp = {}

    for sim_dir in sim_dirs:
        # Extract T using regex
        match = re.search(r"T_(\d+\.\d+)", os.path.basename(sim_dir))
        if not match:
            continue

        T = float(match.group(1))
        energy_path = os.path.join(sim_dir, "ensemble_of_energies.npy")

        if os.path.exists(energy_path):
            energies = np.load(energy_path)
            data_by_temp[T] = energies

    if not data_by_temp:
        print(f"No se encontraron archivos de energía para {protein}.")
        return

    sorted_temps = sorted(data_by_temp.keys())
    mean_energies = []
    cv_values = []

    for T in sorted_temps:
        energies = data_by_temp[T]
        mean_E = np.mean(energies)
        var_E = np.var(energies)
        cv = var_E / (T**2)

        mean_energies.append(mean_E)
        cv_values.append(cv)

    # --- Plotting ---
    print(f"Generando gráficos térmicos para {protein}...")

    # 1. Thermal Properties (Energy & Cv)
    thermal_props_path = os.path.join(results_base, "thermal_properties_vs_T")
    plots.plot_thermal_properties(sorted_temps, mean_energies, cv_values, thermal_props_path)

    # 2. KDE Distributions
    kde_distributions_path = os.path.join(results_base, "energy_distributions_vs_T")
    plots.plot_thermal_energy_distributions(data_by_temp, kde_distributions_path)

    # 3. Save numerical results
    np.savez(
        os.path.join(results_base, "thermal_analysis_summary.npz"),
        temperatures=sorted_temps,
        mean_energies=mean_energies,
        cv_values=cv_values
    )

    print(f"Análisis térmico completado para {protein}. Resultados en {results_base}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Tenés que pasarle el nombre de la familia. Ej: python analyze_thermal_ensembles.py ubiquitin")
        sys.exit(1)

    protein = sys.argv[1]
    analyze_protein_thermal(protein)
