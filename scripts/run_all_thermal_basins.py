import os
import subprocess
import glob
import sys

def main():
    # Check for local execution flag
    local_run = "--local" in sys.argv or "-l" in sys.argv

    # Base directories relative to repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(repo_root, "scripts")
    results_dir_base = os.path.join(repo_root, "results")
    template_path = os.path.join(scripts_dir, "slurm_thermal_basins.sh")

    if not os.path.exists(template_path):
        print(f"Error: Template {template_path} not found.")
        return

    # Find all families in data that have potts.npz
    data_dir_base = os.path.join(repo_root, "data")
    if not os.path.exists(data_dir_base):
        print(f"Error: Data directory {data_dir_base} not found.")
        return

    families = sorted([d for d in os.listdir(data_dir_base) if os.path.isdir(os.path.join(data_dir_base, d))])

    active_families = []
    for protein in families:
        potts_path = os.path.join(data_dir_base, protein, "potts.npz")
        if os.path.exists(potts_path):
            active_families.append(protein)

    print(f"Encontradas {len(active_families)} familias con Potts model.")

    for protein in active_families:
        thermal_res_path = os.path.join(results_dir_base, protein, "thermal_ensembles")

        # Check if thermal ensembles actually exist for this family
        if not (os.path.exists(thermal_res_path) and glob.glob(os.path.join(thermal_res_path, "simulation_of_ensemble_T_*"))):
             print(f"La familia {protein} no tiene simulaciones térmicas calculadas aún. Salteando.")
             continue

        if local_run:
            print(f"\n[LOCAL] Iniciando análisis de cuencas térmicas local para {protein}...")
            try:
                # Run local sequential execution of run_thermal_basins.py
                result = subprocess.run(["python", "run_thermal_basins.py", protein], cwd=scripts_dir)
                if result.returncode == 0:
                    print(f"[LOCAL] Análisis completado con éxito para {protein}!")
                else:
                    print(f"[LOCAL] Error durante el análisis para {protein} (código de salida: {result.returncode}).")
            except Exception as e:
                print(f"[LOCAL] Error ejecutando análisis para {protein}: {e}")
        else:
            print(f"Preparando job de cuencas térmicas para {protein}...")

            protein_results_dir = os.path.join(results_dir_base, protein)
            os.makedirs(protein_results_dir, exist_ok=True)

            with open(template_path, 'r') as f:
                job_script_content = f.read().replace("{PROTEIN}", protein)

            job_script_path = os.path.join(protein_results_dir, f"submit_thermal_basins_{protein}.sh")

            with open(job_script_path, 'w') as f:
                f.write(job_script_content)

            # Submit job via sbatch
            try:
                result = subprocess.run(["sbatch", job_script_path], cwd=scripts_dir, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"Job enviado para {protein}: {result.stdout.strip()}")
                else:
                    print(f"Error enviando job para {protein}: {result.stderr.strip()}")
                    print(f"--> Tip: Si el controlador de SLURM está caído o no responde, podés correr el script de forma local usando: python run_all_thermal_basins.py --local")
            except FileNotFoundError:
                print(f"Comando 'sbatch' no encontrado. Job guardado en {job_script_path} (no enviado).")
                print(f"--> Tip: Podés ejecutar el análisis localmente con: python run_all_thermal_basins.py --local")

if __name__ == "__main__":
    main()
