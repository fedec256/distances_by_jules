import os
import subprocess

def main():
    # Base directories relative to repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    domains_dir = os.path.join(repo_root, "domains")
    data_dir_base = os.path.join(repo_root, "data")
    results_dir_base = os.path.join(repo_root, "results")
    scripts_dir = os.path.join(repo_root, "scripts")

    template_path = os.path.join(scripts_dir, "slurm_neff_sampled.sh")

    if not os.path.exists(domains_dir):
        print(f"Error: Domains directory {domains_dir} not found.")
        return

    families = sorted([d for d in os.listdir(domains_dir) if os.path.isdir(os.path.join(domains_dir, d))])
    print(f"Encontradas {len(families)} familias en {domains_dir}")

    for protein in families:
        print(f"\nProcessing family: {protein}")

        # Ensure symlinks in data/
        protein_data_dir = os.path.join(data_dir_base, protein)
        protein_domains_dir = os.path.join(domains_dir, protein)

        os.makedirs(data_dir_base, exist_ok=True)
        if not os.path.exists(protein_data_dir):
            try:
                os.symlink(protein_domains_dir, protein_data_dir)
                print(f"Created symlink: {protein_data_dir} -> {protein_domains_dir}")
            except Exception as e:
                print(f"Could not create symlink for {protein}: {e}")

        # Ensure results directory
        protein_results_dir = os.path.join(results_dir_base, protein)
        os.makedirs(protein_results_dir, exist_ok=True)

        # Generate SLURM submission script from template
        with open(template_path, 'r') as f:
            job_script_content = f.read().replace("{PROTEIN}", protein)

        job_script_path = os.path.join(protein_results_dir, f"submit_neff_sampled_{protein}.sh")
        with open(job_script_path, 'w') as f:
            f.write(job_script_content)

        # Submit job to SLURM cluster
        try:
            result = subprocess.run(["sbatch", job_script_path], cwd=scripts_dir, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Submitted job for {protein}: {result.stdout.strip()}")
            else:
                print(f"Error submitting job for {protein}: {result.stderr.strip()}")
        except FileNotFoundError:
            print(f"Command 'sbatch' not found. Generated script saved to {job_script_path}")

if __name__ == "__main__":
    main()
