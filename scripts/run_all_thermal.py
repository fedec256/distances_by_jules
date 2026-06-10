import os
import subprocess
import glob

def main():
    # Base directories relative to repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(repo_root, "scripts")
    results_dir_base = os.path.join(repo_root, "results")
    template_path = os.path.join(scripts_dir, "slurm_thermal_ensembles.sh")

    if not os.path.exists(template_path):
        print(f"Error: Template {template_path} not found.")
        return

    # Find all families in data that have potts.npz
    data_dir_base = os.path.join(repo_root, "data")
    families = sorted([d for d in os.listdir(data_dir_base) if os.path.isdir(os.path.join(data_dir_base, d))])

    active_families = []
    for protein in families:
        potts_path = os.path.join(data_dir_base, protein, "potts.npz")
        if os.path.exists(potts_path):
            active_families.append(protein)

    print(f"Encontradas {len(active_families)} familias listas para simulaciones térmicas.")

    for protein in active_families:
        print(f"Preparando job térmico para {protein}...")

        protein_results_dir = os.path.join(results_dir_base, protein)
        os.makedirs(protein_results_dir, exist_ok=True)

        with open(template_path, 'r') as f:
            job_script_content = f.read().replace("{PROTEIN}", protein)

        job_script_path = os.path.join(protein_results_dir, f"submit_thermal_{protein}.sh")

        with open(job_script_path, 'w') as f:
            f.write(job_script_content)

        # Submit job
        try:
            result = subprocess.run(["sbatch", job_script_path], cwd=scripts_dir, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Job enviado para {protein}: {result.stdout.strip()}")
            else:
                print(f"Error enviando job para {protein}: {result.stderr.strip()}")
        except FileNotFoundError:
            print(f"Comando 'sbatch' no encontrado. Job guardado en {job_script_path}.")

if __name__ == "__main__":
    main()
