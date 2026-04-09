import numpy as np
import os
import sys
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import glob

sys.path.append('../modules')
import alignments_functions

if len(sys.argv) < 2:
    print("Error: Tenés que pasarle el nombre de la familia. Ej: python compute_distances.py ubiquitin")
    sys.exit(1)

protein = sys.argv[1]

data_dir = f"../data/{protein}"
msa_path = os.path.join(data_dir, "MSA.fasta")
potts_path  = os.path.join(data_dir, "potts.npz")
w_path =os.path.join(data_dir, "weights.npy")
simulations_path = f"../results/{protein}/simulations_of_frozen_alignments/"

search_pattern = os.path.join(simulations_path, "frozen_alignment_*")
simulation_folders = glob.glob(search_pattern)

if not simulation_folders:
    print(f"No se encontraron simulaciones en {simulations_path}")
    sys.exit(0)

potts_model = np.load(potts_path)
h = potts_model["h"]
J = potts_model["J"]

for data_path in simulation_folders:
    print(f"Procesando: {data_path}")
    
    # Archivos de salida
    out_hi = os.path.join(data_path, "hi_distance.npy")
    out_Z_hi = os.path.join(data_path, "hi_Z_linkage_average.npy")
    out_order_hi = os.path.join(data_path, "hi_orderZ.npy")

    out_hamming = os.path.join(data_path, "hamming_distance.npy")
    out_Z_hamming = os.path.join(data_path, "hamming_Z_linkage_average.npy")
    out_order_hamming = os.path.join(data_path, "hamming_orderZ.npy")


    
    # Chequeo de cosas calculadas para saltear
    if os.path.exists(out_hi) and os.path.exists(out_hamming) and os.path.exists(out_order_hi) and os.path.exists(out_order_hamming):
        print(f"-> Distancias y clustering ya calculados. Salteando...\n")
        continue

    # Si no existen, cargo secuencias y energías y computo cosas
    path_e = os.path.join(data_path, 'frozen_energies.npy')
    path_a = os.path.join(data_path, 'frozen_alignment.npy')
    
    try:
        frozen_energies = np.load(path_e)
        frozen_sequences = np.load(path_a)
    except FileNotFoundError:
        print(f"-> Faltan archivos .npy base en esta carpeta. Salteando...\n")
        continue

    print(" -> Calculando distancias...")
    hi_distance = alignments_functions.distance_in_hi(frozen_sequences, potts_path)
    hamming_d_condensed = pdist(frozen_sequences, "hamming")
    hamming_distance_square = squareform(hamming_d_condensed)

    print("-> Calculando clustering jerárquico...")
    hi_d_condensed = squareform(hi_distance)
    hi_Z_linkage = linkage(hi_d_condensed, method='average')
    hi_orderZ = leaves_list(hi_Z_linkage)

#    hamming_d_condensed = squareform(hamming_distance)
    hamming_Z_linkage = linkage(hamming_d_condensed, method="average")
    hamming_orderZ = leaves_list(hamming_Z_linkage)

    print("  -> Guardando resultados...")
    np.save(out_hi, hi_distance)
    np.save(out_Z_hi, hi_Z_linkage)
    np.save(out_order_hi, hi_orderZ)

    np.save(out_hamming, hamming_distance_square)
    np.save(out_Z_hamming, hamming_Z_linkage)
    np.save(out_order_hamming, hamming_orderZ)

    
    print("-> ¡Listo!\n")

print("Proceso finalizado para todas las carpetas.")