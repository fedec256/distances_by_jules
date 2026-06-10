import numpy as np
import os
import sys
import glob

sys.path.append('../modules')
import alignments_functions

def find_msa_file(domain_path):
    for pattern in ["MSA.fasta", "MSA_nogap.fasta", "*.fasta"]:
        matches = glob.glob(os.path.join(domain_path, pattern))
        if matches:
            return matches[0]
    return None

def main():
    domains_dir = "../domains"
    families = sorted([d for d in os.listdir(domains_dir) if os.path.isdir(os.path.join(domains_dir, d))])

    print(f"{'Family':<20} | {'N':<10} | {'Neff':<10}")
    print("-" * 45)

    for protein in families:
        domain_path = os.path.join(domains_dir, protein)
        msa_path = find_msa_file(domain_path)
        if not msa_path:
            continue

        seqs, _ = alignments_functions.load_msa(msa_path)
        N = len(seqs)

        # Try to load existing weights
        w_path = os.path.join(domain_path, "weights.npy")
        if os.path.exists(w_path):
            weights = np.load(w_path)
        else:
            # We don't want to spend too much time here, so we only compute if N is small
            # or just skip if not present for now, or use a subset to estimate.
            # Actually, compute_sequences_weight_for_many_sequences is fast enough with numba.
            # But let's check if it exists in data/ too
            data_w_path = os.path.join("../data", protein, "weights.npy")
            if os.path.exists(data_w_path):
                weights = np.load(data_w_path)
            else:
                # If N is huge, maybe don't compute right now or just say "TBD"
                if N > 20000:
                    weights = None
                else:
                    MSA_np = alignments_functions.MSA_to_numpy(np.array([[c for c in str(s)] for s in seqs]))
                    weights = alignments_functions.compute_sequences_weight_for_many_sequences(MSA_np)

        if weights is not None:
            neff = np.sum(weights)
            print(f"{protein:<20} | {N:<10} | {neff:<10.2f}")
        else:
            print(f"{protein:<20} | {N:<10} | {'Large (TBD)':<10}")

if __name__ == "__main__":
    main()
