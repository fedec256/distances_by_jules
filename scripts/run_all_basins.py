import os
import subprocess
import glob

def main():
    # Base directories relative to repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(repo_root, "scripts")
    results_dir_base = os.path.join(repo_root, "results")
    template_path = os.path.join(scripts_dir, "slurm_basin_analysis.sh")

    if not os.path.exists(template_path):
        print(f"Error: Template {template_path} not found.")
        return

    # Find all families in results that have simulations
    results_folders = glob.glob(os.path.join(results_dir_base, "*"))
    families = [os.path.basename(f) for f in results_folders if os.path.isdir(f)]

    active_families = []
    for protein in families:
        sim_path = os.path.join(results_dir_base, protein, "simulations_of_frozen_alignments")
        if os.path.exists(sim_path) and glob.glob(os.path.join(sim_path, "frozen_alignment_*")):
            active_families.append(protein)

    print(f"Encontradas {len(active_families)} familias con resultados para analizar.")

    for protein in sorted(active_families):
        print(f"Preparando job de cuencas para {protein}...")

        protein_results_dir = os.path.join(results_dir_base, protein)

        # We use the existing slurm_basin_analysis.sh template
        with open(template_path, 'r') as f:
            job_script_content = f.read().replace("{PROTEIN}", protein)

        job_script_path = os.path.join(protein_results_dir, f"submit_basin_{protein}.sh")

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
