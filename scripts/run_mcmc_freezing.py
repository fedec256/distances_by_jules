if __name__ == "__main__":

    import numpy as np
    import os
    import sys

    sys.path.append('../modules')
    import alignments_functions
    import mcmc_functions as mcmc

    if len(sys.argv) < 2:
        print("Error: Tenés que pasarle el nombre de la familia. Ej: python compute_distances.py ubiquitin")
        sys.exit(1)

    protein = sys.argv[1]
    nsteps = int(sys.argv[2])
    msa_path = f"../data/{protein}/MSA.fasta"
    potts_path = f"../data/{protein}/potts.npz"
    simulations_path = f"../results/{protein}/simulations_of_frozen_alignments/"
    os.makedirs(simulations_path, exist_ok=True)

    seqs, names = alignments_functions.load_msa(msa_path)

    potts_model = np.load(potts_path)
    h = potts_model["h"]
    J = potts_model["J"]

    MSA=np.empty([len(seqs),len(seqs[0])],dtype='<U1')
    for j in range(len(seqs)):
        MSA[j,:]=[i for i in seqs[j]]

    MSA_np = alignments_functions.MSA_to_numpy(MSA)

    mcmc.freezing_alignment(simulations_path, MSA_np, nsteps, h, J)