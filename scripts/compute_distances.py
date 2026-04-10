import numpy as np
import os
import sys
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import glob

sys.path.append('../modules')
import alignments_functions

def run_analysis(frozen_sequences, frozen_energies, data_path, potts_path, suffix=""):
    # Output filenames
    out_hi = os.path.join(data_path, f"hi_distance{suffix}.npy")
    out_Z_hi = os.path.join(data_path, f"hi_Z_linkage_average{suffix}.npy")
    out_order_hi = os.path.join(data_path, f"hi_orderZ{suffix}.npy")

    out_hamming = os.path.join(data_path, f"hamming_distance{suffix}.npy")
    out_Z_hamming = os.path.join(data_path, f"hamming_Z_linkage_average{suffix}.npy")
    out_order_hamming = os.path.join(data_path, f"hamming_orderZ{suffix}.npy")

    print(f" -> Calculando distancias {suffix}...")
    hi_distance = alignments_functions.distance_in_hi(frozen_sequences, potts_path).astype(np.float32)
    
    # Use pdist for Hamming, then squareform
    hamming_d_condensed = pdist(frozen_sequences, "hamming").astype(np.float32)
    hamming_distance_square = squareform(hamming_d_condensed).astype(np.float32)

    print(f" -> Calculando clustering jerárquico {suffix}...")
    # hierarchical clustering needs condensed distance matrix
    hi_d_condensed = squareform(hi_distance, force='tovector')
    hi_Z_linkage = linkage(hi_d_condensed, method='average')
    hi_orderZ = leaves_list(hi_Z_linkage)

    hamming_Z_linkage = linkage(hamming_d_condensed, method="average")
    hamming_orderZ = leaves_list(hamming_Z_linkage)

    print(f"  -> Guardando resultados {suffix}...")
    np.save(out_hi, hi_distance)
    np.save(out_Z_hi, hi_Z_linkage)
    np.save(out_order_hi, hi_orderZ)

    np.save(out_hamming, hamming_distance_square)
    np.save(out_Z_hamming, hamming_Z_linkage)
    np.save(out_order_hamming, hamming_orderZ)
    
    # Save energies for sampled subset as well
    if suffix:
        np.save(os.path.join(data_path, f"frozen_energies{suffix}.npy"), frozen_energies)
        np.save(os.path.join(data_path, f"frozen_alignment{suffix}.npy"), frozen_sequences)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Tenés que pasarle el nombre de la familia. Ej: python compute_distances.py ubiquitin")
        sys.exit(1)

    protein = sys.argv[1]

    data_dir = f"../data/{protein}"
    msa_path = os.path.join(data_dir, "MSA.fasta")
    potts_path = os.path.join(data_dir, "potts.npz")
    w_path = os.path.join(data_dir, "weights.npy")
    simulations_path = f"../results/{protein}/simulations_of_frozen_alignments/"

    search_pattern = os.path.join(simulations_path, "frozen_alignment_*")
    simulation_folders = glob.glob(search_pattern)

    if not simulation_folders:
        print(f"No se encontraron simulaciones en {simulations_path}")
        sys.exit(0)

    # Load weights if available for sampling
    weights = None
    if os.path.exists(w_path):
        weights = np.load(w_path)
    else:
        print(f"Warning: No weights.npy found at {w_path}")

    for data_path in simulation_folders:
        print(f"Procesando: {data_path}")

        path_e = os.path.join(data_path, 'frozen_energies.npy')
        path_a = os.path.join(data_path, 'frozen_alignment.npy')

        try:
            frozen_energies = np.load(path_e)
            frozen_sequences = np.load(path_a)
        except FileNotFoundError:
            print(f"-> Faltan archivos .npy base en esta carpeta. Salteando...\n")
            continue

        n_seqs = len(frozen_sequences)
        print(f"Total secuencias: {n_seqs}")

        # Thresholds
        MAX_FULL = 20000
        N_SAMPLE = 5000

        # Run Full if small enough
        if n_seqs <= MAX_FULL:
            run_analysis(frozen_sequences, frozen_energies, data_path, potts_path, suffix="")
        else:
            print(f"-> Demasiadas secuencias ({n_seqs}) para análisis completo. Solo muestreo.")

        # Always run Sampling if n_seqs > N_SAMPLE
        if n_seqs > N_SAMPLE:
            print(f"-> Realizando muestreo de {N_SAMPLE} secuencias...")
            if weights is not None and len(weights) == n_seqs:
                prob = weights / weights.sum()
                indices = np.random.choice(n_seqs, N_SAMPLE, p=prob, replace=False)
            else:
                print("-> Usando muestreo uniforme (sin pesos).")
                indices = np.random.choice(n_seqs, N_SAMPLE, replace=False)

            sampled_seqs = frozen_sequences[indices]
            sampled_energies = frozen_energies[indices]
            run_analysis(sampled_seqs, sampled_energies, data_path, potts_path, suffix="_sampled")

        print("-> ¡Listo!\n")

    print("Proceso finalizado para todas las carpetas.")
