import numpy as np
import os
import sys
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import glob

sys.path.append('../modules')
import alignments_functions

def run_analysis(frozen_sequences, frozen_energies, frozen_weights, data_path, potts_path, suffix=""):
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
    
    # Save energies/sequences/weights for the set used (especially if sampled)
    np.save(os.path.join(data_path, f"frozen_energies{suffix}.npy"), frozen_energies)
    np.save(os.path.join(data_path, f"frozen_alignment{suffix}.npy"), frozen_sequences)
    if frozen_weights is not None:
        np.save(os.path.join(data_path, f"frozen_weights{suffix}.npy"), frozen_weights)


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
        MAX_ANALYSIS = 20000
        N_SAMPLE = 5000

        # --- "Full" or "Maximized" Analysis ---
        if n_seqs <= MAX_ANALYSIS:
            print(f"-> Ejecutando análisis completo ({n_seqs} secuencias).")
            run_analysis(frozen_sequences, frozen_energies, weights, data_path, potts_path, suffix="")
        else:
            print(f"-> Muestreando {MAX_ANALYSIS} de {n_seqs} para análisis principal.")
            if weights is not None and len(weights) == n_seqs:
                prob = weights / weights.sum()
                indices = np.random.choice(n_seqs, MAX_ANALYSIS, p=prob, replace=False)
                run_analysis(frozen_sequences[indices], frozen_energies[indices], weights[indices], data_path, potts_path, suffix="")
            else:
                indices = np.random.choice(n_seqs, MAX_ANALYSIS, replace=False)
                run_analysis(frozen_sequences[indices], frozen_energies[indices], None, data_path, potts_path, suffix="")

        # --- "Lightweight" Sampling (if enough sequences) ---
        if n_seqs > N_SAMPLE:
            print(f"-> Realizando muestreo liviano de {N_SAMPLE} secuencias para comparación rápida...")
            if weights is not None and len(weights) == n_seqs:
                prob = weights / weights.sum()
                indices_small = np.random.choice(n_seqs, N_SAMPLE, p=prob, replace=False)
                run_analysis(frozen_sequences[indices_small], frozen_energies[indices_small], weights[indices_small], data_path, potts_path, suffix="_sampled")
            else:
                indices_small = np.random.choice(n_seqs, N_SAMPLE, replace=False)
                run_analysis(frozen_sequences[indices_small], frozen_energies[indices_small], None, data_path, potts_path, suffix="_sampled")

        print("-> ¡Listo!\n")

    print("Proceso finalizado para todas las carpetas.")
