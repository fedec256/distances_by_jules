import os
import subprocess
import glob

def find_msa_file(domain_path):
    """Finds the MSA file in the domain directory."""
    for pattern in ["MSA.fasta", "MSA_nogap.fasta", "*.fasta"]:
        matches = glob.glob(os.path.join(domain_path, pattern))
        if matches:
            return matches[0]
    return None

def main():
    # Base directories relative to repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    domains_dir = os.path.join(repo_root, "domains")
    scripts_dir = os.path.join(repo_root, "scripts")
    data_dir_base = os.path.join(repo_root, "data")
    results_dir_base = os.path.join(repo_root, "results")

    template_path = os.path.join(scripts_dir, "slurm_template.sh")

    if not os.path.exists(template_path):
        print(f"Error: Template {template_path} not found.")
        return

    with open(template_path, 'r') as f:
        template = f.read()

    # Get all potential protein families
    if not os.path.exists(domains_dir):
        print(f"Error: Domains directory {domains_dir} not found.")
        return

    families = sorted([d for d in os.listdir(domains_dir) if os.path.isdir(os.path.join(domains_dir, d))])

    print(f"Encontradas {len(families)} familias en {domains_dir}")

    for protein in families:
        domain_path = os.path.join(domains_dir, protein)

        # Link domains to data folder so scripts can find them
        protein_data_dir = os.path.join(data_dir_base, protein)
        os.makedirs(data_dir_base, exist_ok=True)
        if not os.path.exists(protein_data_dir):
            try:
                os.symlink(os.path.abspath(domain_path), protein_data_dir)
                print(f"Linked {domain_path} -> {protein_data_dir}")
            except Exception as e:
                print(f"Could not link {domain_path}: {e}")

        msa_path = find_msa_file(domain_path)
        potts_path = os.path.join(domain_path, "potts.npz")

        # Check if basic data exists
        if not (msa_path and os.path.exists(potts_path)):
            print(f"Skipping {protein}: Missing MSA (*.fasta) or potts.npz")
            continue

        print(f"Preparando job para {protein}...")

        # Create results dir
        protein_results_dir = os.path.join(results_dir_base, protein)
        os.makedirs(protein_results_dir, exist_ok=True)

        # Customize template
        job_script_content = template.replace("{PROTEIN}", protein)
        job_script_path = os.path.join(protein_results_dir, f"submit_{protein}.sh")

        with open(job_script_path, 'w') as f:
            f.write(job_script_content)

        # Submit job
        try:
            # We run sbatch from the scripts directory
            result = subprocess.run(["sbatch", job_script_path], cwd=scripts_dir, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Job enviado para {protein}: {result.stdout.strip()}")
            else:
                print(f"Error enviando job para {protein}: {result.stderr.strip()}")
        except FileNotFoundError:
            print(f"Comando 'sbatch' no encontrado. Job para {protein} guardado en {job_script_path} (no enviado).")

if __name__ == "__main__":
    main()
