import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def distance_and_energy_of_sequences_graph(distance_matrix, orderZ, energies, saving_path=None):
    """
    Plots a heatmap of the distance matrix ordered by clustering,
    with a corresponding energy plot below it.
    Optimized for large matrices using imshow and rasterization.
    """
    D_clusteredZ = distance_matrix[orderZ][:, orderZ]
    E_clusteredZ = energies[orderZ]

    nseq = len(E_clusteredZ)
    step = max(1, nseq // 10)

    xticks = np.arange(0, nseq, step)

    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)

    # --- Heatmap ---
    ax1 = fig.add_subplot(gs[0])

    # Use imshow with rasterization for performance with many sequences
    # Set vmin=0 and vmax=1 to fix color scale for Hamming distances
    im = ax1.imshow(
        D_clusteredZ,
        cmap='viridis',
        aspect='auto',
        rasterized=True,
        interpolation='nearest',
        vmin=0.0,
        vmax=1.0
    )

    ax1.set_title("Hamming Distances (Ordered by Clustering)", fontsize=12, pad=8)
    ax1.set_xlim(0, nseq)
    ax1.set_ylabel("Sequences (Hierarchical Clustering Order)", fontsize=11)
    ax1.set_yticks(xticks)
    ax1.set_yticklabels(xticks + 1)
    # Hide x-ticks for the top plot as they are shared with the bottom one
    ax1.tick_params(labelbottom=False)

    # --- Energy Plot ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(np.arange(nseq), E_clusteredZ, color='black', lw=1)
    ax2.set_ylabel("DCA Energy", fontsize=11)
    ax2.set_xlabel("Sequences", fontsize=11)

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks + 1)
    ax2.tick_params(axis='x', labelrotation=0)

    # Add a colorbar for the heatmap
    # plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Adjust margins
    plt.subplots_adjust(left=0.15, right=0.90, top=0.95, bottom=0.10, hspace=0.05)

    if saving_path is not None:
        plt.savefig(f"{saving_path}.png", dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(f"{saving_path}.pdf", dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()

    # Close fig to free memory
    plt.close(fig)

def plot_basin_size_vs_rank(counts, saving_path=None):
    """
    Plots Basin Size vs Rank.
    """
    ranks = np.arange(1, len(counts) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Linear plot
    ax1.plot(ranks, counts, marker='o', linestyle='-', markersize=4, color='teal')
    ax1.set_title("Basin Size vs Rank (Linear)")
    ax1.set_xlabel("Rank")
    ax1.set_ylabel("Basin Size (Number of Sequences)")
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    # Log-log plot
    ax2.loglog(ranks, counts, marker='o', linestyle='-', markersize=4, color='maroon')
    ax2.set_title("Basin Size vs Rank (Log-Log)")
    ax2.set_xlabel("Rank (log)")
    ax2.set_ylabel("Basin Size (log)")
    ax2.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()

    if saving_path:
        plt.savefig(f"{saving_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{saving_path}.pdf", dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)

def plot_energy_vs_rank(energies, saving_path=None):
    """
    Plots Energy of basins vs Rank.
    """
    ranks = np.arange(1, len(energies) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(ranks, energies, marker='o', linestyle='-', markersize=4, color='darkblue')
    ax.set_title("Basin Energy vs Rank")
    ax.set_xlabel("Rank")
    ax.set_ylabel("DCA Energy")
    ax.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()

    if saving_path:
        plt.savefig(f"{saving_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{saving_path}.pdf", dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)

def distance_basin_and_energy_graph(distance_matrix, orderZ, energies, basin_sizes, saving_path=None):
    """
    3-panel graph:
    - Top: Basin Size
    - Middle: Hamming Heatmap
    - Bottom: Energy
    """
    D_clusteredZ = distance_matrix[orderZ][:, orderZ]
    E_clusteredZ = energies[orderZ]
    B_clusteredZ = basin_sizes[orderZ]

    nseq = len(E_clusteredZ)
    step = max(1, nseq // 10)
    xticks = np.arange(0, nseq, step)

    fig = plt.figure(figsize=(8, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 4, 1], hspace=0.05)

    # --- Top: Basin Size ---
    ax0 = fig.add_subplot(gs[0])
    ax0.bar(np.arange(nseq), B_clusteredZ, color='teal', width=1.0)
    ax0.set_ylabel("Basin Size", fontsize=11)
    ax0.set_title("Distance Matrix with Basin Size and Energy", fontsize=12, pad=10)
    ax0.tick_params(labelbottom=False)

    # --- Middle: Heatmap ---
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    # Set vmin=0 and vmax=1 to fix color scale for Hamming distances
    im = ax1.imshow(
        D_clusteredZ,
        cmap='viridis',
        aspect='auto',
        rasterized=True,
        interpolation='nearest',
        vmin=0.0,
        vmax=1.0
    )
    ax1.set_ylabel("Sequences (Clustered)", fontsize=11)
    ax1.set_yticks(xticks)
    ax1.set_yticklabels(xticks + 1)
    ax1.tick_params(labelbottom=False)

    # --- Bottom: Energy ---
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax2.plot(np.arange(nseq), E_clusteredZ, color='black', lw=1)
    ax2.set_ylabel("DCA Energy", fontsize=11)
    ax2.set_xlabel("Sequences", fontsize=11)

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks + 1)
    ax2.tick_params(axis='x', labelrotation=0)

    plt.subplots_adjust(left=0.15, right=0.90, top=0.95, bottom=0.08, hspace=0.05)

    if saving_path is not None:
        plt.savefig(f"{saving_path}.png", dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(f"{saving_path}.pdf", dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()

    plt.close(fig)

def plot_thermal_properties(temps, mean_energies, cv_values, saving_path=None):
    """
    Plots Mean Energy and Specific Heat (Cv) vs Temperature.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # --- Mean Energy vs T ---
    ax1.plot(temps, mean_energies, marker='o', color='navy', linestyle='-', linewidth=2)
    ax1.set_ylabel("Mean DCA Energy", fontsize=12)
    ax1.set_title("Thermal Properties", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # --- Cv vs T ---
    ax2.plot(temps, cv_values, marker='s', color='crimson', linestyle='-', linewidth=2)
    ax2.set_ylabel("Specific Heat (Cv)", fontsize=12)
    ax2.set_xlabel("Temperature (T)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if saving_path:
        plt.savefig(f"{saving_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{saving_path}.pdf", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def plot_thermal_energy_distributions(temp_energy_dict, saving_path=None):
    """
    Combined KDE plot of energy distributions for multiple temperatures.
    """
    plt.figure(figsize=(10, 6))

    sorted_temps = sorted(temp_energy_dict.keys())
    # Create a colormap
    norm = plt.Normalize(min(sorted_temps), max(sorted_temps))
    cmap = plt.get_cmap("viridis")

    for T in sorted_temps:
        energies = temp_energy_dict[T]
        sns.kdeplot(energies, color=cmap(norm(T)), label=f"T={T:.2f}", fill=False, alpha=0.7)

    plt.title("Energy Distributions across Temperatures", fontsize=14)
    plt.xlabel("DCA Energy", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    # Legend can be huge, so we put a colorbar instead if many temps
    if len(sorted_temps) > 10:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label("Temperature")
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if saving_path:
        plt.savefig(f"{saving_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{saving_path}.pdf", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def distance_and_energy_of_sequences_graph_with_seaborn(distance_matrix, orderZ, energies, saving_path = None):
    """Legacy version using seaborn. Less efficient for very large matrices."""
    D_clusteredZ = distance_matrix[orderZ][:, orderZ]
    E_clusteredZ = energies[orderZ]

    nseq = len(E_clusteredZ)
    step = max(1, nseq // 10)
    xticks = np.arange(0, nseq, step)

    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(
        D_clusteredZ,
        cmap='viridis',
        ax=ax1,
        cbar=True,
    )
    ax1.set_title("Hamming Distances (Clustered)", fontsize=12)
    ax1.set_yticks(xticks)
    ax1.set_yticklabels(xticks + 1)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(np.arange(nseq), E_clusteredZ, color='black', lw=1)
    ax2.set_ylabel("Energy")

    if saving_path is not None:
        plt.savefig(f"{saving_path}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)
