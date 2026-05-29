import numpy as np
import os
import sys
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import glob

sys.path.append('../modules')
import alignments_functions

def run_hamming_analysis(frozen_sequences, frozen_energies, frozen_weights, sampled_indices, names, data_path, suffix=""):
    """Computes Hamming distances and clustering for a subset of sequences."""

    # Output filenames
    out_hamming = os.path.join(data_path, f"hamming_distance{suffix}.npy")
    out_Z_hamming = os.path.join(data_path, f"hamming_Z_linkage_average{suffix}.npy")
    out_order_hamming = os.path.join(data_path, f"hamming_orderZ{suffix}.npy")
    out_indices = os.path.join(data_path, f"sampled_indices{suffix}.npy")
    out_names = os.path.join(data_path, f"sampled_names{suffix}.txt")

    print(f" -> Calculando distancias Hamming {suffix} ({len(frozen_sequences)} secuencias)...")
    
    # Use pdist for Hamming (efficiently computes condensed matrix)
    hamming_d_condensed = pdist(frozen_sequences, "hamming").astype(np.float32)
    hamming_distance_square = squareform(hamming_d_condensed).astype(np.float32)

    print(f" -> Calculando clustering jerárquico {suffix}...")
    hamming_Z_linkage = linkage(hamming_d_condensed, method="average")
    hamming_orderZ = leaves_list(hamming_Z_linkage)

    print(f"  -> Guardando resultados {suffix}...")
    np.save(out_hamming, hamming_distance_square)
    np.save(out_Z_hamming, hamming_Z_linkage)
    np.save(out_order_hamming, hamming_orderZ)
    np.save(out_indices, sampled_indices)
    
    with open(out_names, "w") as f:
        for idx in sampled_indices:
            f.write(f"{names[idx]}\n")

    # Save energies/sequences/weights for the set used
    np.save(os.path.join(data_path, f"frozen_energies{suffix}.npy"), frozen_energies)
    np.save(os.path.join(data_path, f"frozen_alignment{suffix}.npy"), frozen_sequences)
    if frozen_weights is not None:
        np.save(os.path.join(data_path, f"frozen_weights{suffix}.npy"), frozen_weights)

def find_msa_file(protein_data_dir):
    """Finds the MSA file in the protein data directory."""
    for pattern in ["MSA.fasta", "MSA_nogap.fasta", "*.fasta"]:
        matches = glob.glob(os.path.join(protein_data_dir, pattern))
        if matches:
            return matches[0]
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Tenés que pasarle el nombre de la familia. Ej: python compute_distances.py ubiquitin")
        sys.exit(1)

    protein = sys.argv[1]

    data_dir = f"../data/{protein}"
    w_path = os.path.join(data_dir, "weights.npy")
    simulations_path = f"../results/{protein}/simulations_of_frozen_alignments/"

    msa_path = find_msa_file(data_dir)
    if not msa_path:
        print(f"Error: No se encontró un archivo MSA en {data_dir}")
        sys.exit(1)

    # Load original names from MSA for mapping
    _, names = alignments_functions.load_msa(msa_path)

    search_pattern = os.path.join(simulations_path, "frozen_alignment_*")
    simulation_folders = glob.glob(search_pattern)

    if not simulation_folders:
        print(f"No se encontraron simulaciones en {simulations_path}")
        sys.exit(0)

    for data_path in simulation_folders:
        print(f"Procesando: {data_path}")

        path_e = os.path.join(data_path, 'frozen_energies.npy')
        path_a = os.path.join(data_path, 'frozen_alignment.npy')

        try:
            frozen_energies_all = np.load(path_e)
            frozen_sequences_all = np.load(path_a)
        except FileNotFoundError:
            print(f"-> Faltan archivos .npy base en esta carpeta. Salteando...\n")
            continue

        n_seqs = len(frozen_sequences_all)
        print(f"Total secuencias congeladas: {n_seqs}")

        # 1. Get weights
        if os.path.exists(w_path):
            print(f"Cargando pesos existentes desde {w_path}")
            weights = np.load(w_path)
        else:
            print(f"Pesos no encontrados. Calculando pesos (esto puede tardar para N={n_seqs})...")
            # compute_sequences_weight_for_many_sequences uses jit and parallel
            weights = alignments_functions.compute_sequences_weight_for_many_sequences(frozen_sequences_all)
            np.save(w_path, weights)
            print(f"Pesos guardados en {w_path}")

        # 2. Calculate Neff
        neff = np.sum(weights)
        n_sample = int(round(neff))
        print(f"Neff: {neff:.2f} -> Tamaño de muestra: {n_sample}")

        if n_sample < 2:
             print(f"Warning: n_sample ({n_sample}) es demasiado pequeño. Ajustando a 2.")
             n_sample = min(2, n_seqs)

        # 3. Sample
        print(f"Realizando muestreo de {n_sample} secuencias...")
        prob = weights / weights.sum()
        indices = np.random.choice(n_seqs, n_sample, p=prob, replace=False)
        # Sort indices to maintain some relative order and for consistency
        indices.sort()

        # 4. Run Analysis
        run_hamming_analysis(
            frozen_sequences_all[indices],
            frozen_energies_all[indices],
            weights[indices],
            indices,
            names,
            data_path,
            suffix="_sampled_neff"
        )

        print("-> ¡Listo!\n")

    print("Proceso finalizado para todas las carpetas.")
