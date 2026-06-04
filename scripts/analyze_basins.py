import numpy as np
import os
import sys
import glob

def analyze_basins(frozen_alignment_path):
    print(f"Cargando alineamiento desde {frozen_alignment_path}...")
    # Load alignment
    alignment = np.load(frozen_alignment_path)
    n_seqs, n_pos = alignment.shape
    print(f"Alineamiento cargado: {n_seqs} secuencias, {n_pos} posiciones.")

    # Find unique sequences and their counts
    print("Identificando secuencias únicas y calculando tamaños de cuencas...")
    unique_seqs, counts = np.unique(alignment, axis=0, return_counts=True)

    # Sort by counts in descending order (Rank)
    sort_indices = np.argsort(-counts)
    sorted_counts = counts[sort_indices]
    sorted_unique_seqs = unique_seqs[sort_indices]

    n_basins = len(sorted_counts)
    print(f"Encontradas {n_basins} cuencas únicas.")

    return sorted_counts, sorted_unique_seqs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Tenés que pasarle el nombre de la familia. Ej: python analyze_basins.py ubiquitin")
        sys.exit(1)

    protein = sys.argv[1]
    results_base = f"../results/{protein}/simulations_of_frozen_alignments/"

    # Find all simulation folders
    sim_dirs = glob.glob(os.path.join(results_base, "frozen_alignment_*"))

    if not sim_dirs:
        print(f"No se encontraron simulaciones en {results_base}")
        sys.exit(0)

    for sim_dir in sim_dirs:
        frozen_ali_path = os.path.join(sim_dir, "frozen_alignment.npy")
        if not os.path.exists(frozen_ali_path):
            continue

        print(f"Procesando cuencas en: {sim_dir}")
        counts, unique_seqs = analyze_basins(frozen_ali_path)

        # Save results
        out_path = os.path.join(sim_dir, "basin_analysis.npz")
        np.savez(out_path, counts=counts, unique_sequences=unique_seqs)
        print(f"Resultados guardados en {out_path}")
