import numpy as np
import os
import sys
import glob
from matplotlib import pyplot as plt, colors
import seaborn as sns
sys.path.append('../modules')
import alignments_functions
import mcmc_functions as mcmc
import plots_of_distances as plots

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

seqs, names = alignments_functions.load_msa(msa_path)

MSA=np.empty([len(seqs),len(seqs[0])],dtype='<U1')
for j in range(len(seqs)):
    MSA[j,:]=[i for i in seqs[j]]

MSA_np = alignments_functions.MSA_to_numpy(MSA)

potts_model = np.load(potts_path)
h = potts_model["h"]
J = potts_model["J"]

w = np.load(w_path)

natural_energies = []
for i in range(len(seqs)):
    seq_i = MSA_np[i,:]
    natural_energies.append(mcmc.E_tot(seq_i, h, J))

for data_path in simulation_folders:
    print(f"Procesando: {data_path}")
    
    path_e = os.path.join(data_path, 'frozen_energies.npy')

    out_hi = os.path.join(data_path, "hi_distance.npy")
    out_Z_hi = os.path.join(data_path, "hi_Z_linkage_average.npy")
    out_order_hi = os.path.join(data_path, "hi_orderZ.npy")

    out_hamming = os.path.join(data_path, "hamming_distance.npy")
    out_Z_hamming = os.path.join(data_path, "hamming_Z_linkage_average.npy")
    out_order_hamming = os.path.join(data_path, "hamming_orderZ.npy")

    try:
        frozen_energies = np.load(path_e)

        hi_distance = np.load(out_hi)
        hi_Z_linkage = np.load(out_Z_hi)
        hi_orderZ = np.load(out_order_hi)

        hamming_distance = np.load(out_hamming)
        hamming_Z_linkage = np.load(out_Z_hamming)
        hamming_orderZ = np.load(out_order_hamming)

    except FileNotFoundError:
        print(f"-> Faltan archivos .npy base en esta carpeta. Salteando...\n")
        continue
    
    print("-> Heatmaps de distancias...")
    hi_distances_heatmap_path = os.path.join(data_path, "hi_distances_heatmap")
    plots.distance_and_energy_of_sequences_graph(hi_distance, hi_orderZ, frozen_energies, hi_distances_heatmap_path)

    hamming_distances_heatmap_path = os.path.join(data_path, "hamming_distances_heatmap")
    plots.distance_and_energy_of_sequences_graph(hamming_distance, hamming_orderZ, frozen_energies, hamming_distances_heatmap_path)

    print("-> Graficando distribuciones KDE...")
    kde_plot_path = os.path.join(data_path, "energies_KDE_distribution")
    plt.figure(figsize=(10,6))
    sns.kdeplot(x=frozen_energies , label = "KDEplot Frozen Sequences" , weights = w)
    sns.kdeplot(x=natural_energies, label = "KDEplot Natural Sequences", weights = w)
    plt.legend()
    plt.xlabel("Energy")
    plt.savefig(f"{kde_plot_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{kde_plot_path}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("-> Gráficos listos para esta carpeta.")

print("Ya miramos y graficamos en todas las carpetas!")