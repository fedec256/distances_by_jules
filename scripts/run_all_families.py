import os
import subprocess
import glob

def main():
    domains_dir = "../domains"
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(scripts_dir, "slurm_template.sh")

    if not os.path.exists(template_path):
        print(f"Error: Template {template_path} not found.")
        return

    with open(template_path, 'r') as f:
        template = f.read()

    # Get all potential protein families
    families = [d for d in os.listdir(domains_dir) if os.path.isdir(os.path.join(domains_dir, d))]

    print(f"Encontradas {len(families)} familias en {domains_dir}")

    for protein in families:
        domain_path = os.path.join(domains_dir, protein)
        msa_path = os.path.join(domain_path, "MSA.fasta")
        potts_path = os.path.join(domain_path, "potts.npz")

        # Check if basic data exists
        if not (os.path.exists(msa_path) and os.path.exists(potts_path)):
            print(f"Skipping {protein}: Missing MSA.fasta or potts.npz")
            continue

        print(f"Preparando job para {protein}...")

        # Create results dir
        results_dir = os.path.join("../results", protein)
        os.makedirs(results_dir, exist_ok=True)

        # Customize template
        job_script_content = template.replace("{PROTEIN}", protein)
        job_script_path = os.path.join(results_dir, f"submit_{protein}.sh")

        with open(job_script_path, 'w') as f:
            f.write(job_script_content)

        # Submit job
        try:
            # We run sbatch from the scripts directory because the scripts expect paths relative to ..
            result = subprocess.run(["sbatch", job_script_path], cwd=scripts_dir, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Job enviado para {protein}: {result.stdout.strip()}")
            else:
                print(f"Error enviando job para {protein}: {result.stderr.strip()}")
        except FileNotFoundError:
            print(f"Comando 'sbatch' no encontrado. Job para {protein} guardado en {job_script_path} (no enviado).")

if __name__ == "__main__":
    main()
