import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def distance_and_energy_of_sequences_graph_with_seaborn(distance_matrix, orderZ, energies, saving_path = None):
    D_clusteredZ = distance_matrix[orderZ][:, orderZ]
    E_clusteredZ = energies[orderZ]

    nseq = len(E_clusteredZ)
    step = max(1, nseq // 10)

    xticks = np.arange(0, nseq, step)

    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)

    # --- heatmap ---
    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(
        D_clusteredZ,
        cmap='viridis',
        ax=ax1,
        cbar=False,
    #    xticklabels=False,
    #    yticklabels=False
    )
    ax1.set_title("Distancias ordenadas por clustering jerárquico", fontsize=12, pad=8)
    ax1.set_xlim(0, len(E_clusteredZ))
    ax1.set_aspect("auto")
    ax1.set_ylabel("Secuencias (ordenadas según clustering)", fontsize=11)
    ax1.set_yticks(xticks)
    ax1.set_yticklabels(xticks + 1)

    # --- evo energy plot ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(np.arange(len(E_clusteredZ)), E_clusteredZ, color='black', lw=1)
    ax2.set_ylabel("Evo energy", fontsize=11)
    ax2.set_xlabel("Secuencias (ordenadas según clustering)", fontsize=11)


    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks + 1)
    ax2.tick_params(axis='x', labelrotation=0)

    # --- ajustar márgenes ---
    plt.subplots_adjust(left=0.1, right=0.95, top=0.97, bottom=0.08, hspace=0.05)

    if saving_path is not None:
        plt.savefig(f"{saving_path}.png", dpi=600, bbox_inches='tight', transparent=True)
        plt.savefig(f"{saving_path}.pdf", dpi=600, bbox_inches='tight', transparent=True)

    else:
        plt.show()

    plt.close(fig)

def distance_and_energy_of_sequences_graph(distance_matrix, orderZ, energies, saving_path=None):
    D_clusteredZ = distance_matrix[orderZ][:, orderZ]
    E_clusteredZ = energies[orderZ]

    nseq = len(E_clusteredZ)
    step = max(1, nseq // 10)

    xticks = np.arange(0, nseq, step)

    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)

    # --- heatmap (Versión Optimizada) ---
    ax1 = fig.add_subplot(gs[0])
    
    # imshow y rasterized=True
    im = ax1.imshow(
        D_clusteredZ, 
        cmap='viridis', 
        aspect='auto', 
        rasterized=True, 
        interpolation='nearest' 
    )
    
    ax1.set_title("Distancias ordenadas por clustering jerárquico", fontsize=12, pad=8)
    ax1.set_xlim(0, len(E_clusteredZ))
    ax1.set_ylabel("Secuencias (ordenadas según clustering)", fontsize=11)
    ax1.set_yticks(xticks)
    ax1.set_yticklabels(xticks + 1)

    # --- evo energy plot ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(np.arange(len(E_clusteredZ)), E_clusteredZ, color='black', lw=1)
    ax2.set_ylabel("Evo energy", fontsize=11)
    ax2.set_xlabel("Secuencias (ordenadas según clustering)", fontsize=11)

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks + 1)
    ax2.tick_params(axis='x', labelrotation=0)

    # --- ajustar márgenes ---
    plt.subplots_adjust(left=0.1, right=0.95, top=0.97, bottom=0.08, hspace=0.05)

    if saving_path is not None:
        plt.savefig(f"{saving_path}.png", dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(f"{saving_path}.pdf", dpi=300, bbox_inches='tight', transparent=True)

    else:
        plt.show()
    
    # Fundamental para limpiar la memoria después de cada iteración
    plt.close(fig)