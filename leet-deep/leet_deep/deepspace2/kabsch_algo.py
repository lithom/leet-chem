import numpy as np
import matplotlib.pyplot as plt
import torch
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
import time

from torch.utils.data import Dataset


def kabsch_numpy(P, Q):
    """ Kabsch alignment of two structures P and Q. Returns the aligned version of Q. """
    C = np.matmul(P.transpose((0,2,1)), Q)

    V, S, Wt = np.linalg.svd(C)

    # Correct rotation matrix to ensure the right-handed coordinate system
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
    S[..., -1] = S[..., -1] * (-1) ** d
    V[..., :, -1] = V[..., :, -1] * (-1) ** np.expand_dims(d, axis=-1)

    # Create Rotation matrix U
    U = np.matmul(V, Wt)

    # Rotate P
    Q = np.matmul(Q, U)
    return Q

def kabsch_numpy_2(P, Q):
    """
    Kabsch alignment of two point sets P and Q.
    """
    assert P.shape == Q.shape, "Input shapes must match"
    num_batches = P.shape[0]
    N = P.shape[1]
    Q_aligned = np.zeros((num_batches, N, 3))

    for i in range(num_batches):
        P_centered = P[i] - np.mean(P[i], axis=0)
        Q_centered = Q[i] - np.mean(Q[i], axis=0)

        H = np.dot(P_centered.T, Q_centered)

        U, _, Vt = np.linalg.svd(H)

        d = (np.linalg.det(U) * np.linalg.det(Vt)) < 0.0

        if d:
            U[:, -1] = -U[:, -1]

        R = np.dot(U, Vt)
        Q_aligned[i] = np.dot(Q_centered, R.T)  # rotate Q

    return Q_aligned


def kabsch_numpy_3(P, Q):
    """
    Kabsch alignment of two point sets P and Q.
    """
    assert P.shape == Q.shape, "Input shapes must match"
    num_batches = P.shape[0]
    N = P.shape[1]
    Q_aligned = np.zeros((num_batches, N, 3))

    for i in range(num_batches):
        centroid_P = np.mean(P[i], axis=0)
        centroid_Q = np.mean(Q[i], axis=0)

        P_centered = P[i] - centroid_P
        Q_centered = Q[i] - centroid_Q

        H = np.dot(P_centered.T, Q_centered)

        U, _, Vt = np.linalg.svd(H)

        d = (np.linalg.det(U) * np.linalg.det(Vt)) < 0.0

        if d:
            U[:, -1] = -U[:, -1]

        R = np.dot(U, Vt)
        Q_aligned[i] = np.dot(Q_centered, R.T) + centroid_P  # rotate Q and translate it

    return Q_aligned

def kabsch_numpy_3_single_sample(P, Q):
    """
    Kabsch alignment of two point sets P and Q.
    """
    assert P.shape == Q.shape, "Input shapes must match"
    #num_batches = P.shape[0]
    N = P.shape[0]
    Q_aligned = np.zeros((N, 3))

    #for i in range(num_batches):
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = np.dot(P_centered.T, Q_centered)

    U, _, Vt = np.linalg.svd(H)

    d = (np.linalg.det(U) * np.linalg.det(Vt)) < 0.0

    if d:
        U[:, -1] = -U[:, -1]

    R = np.dot(U, Vt)
    Q_aligned = np.dot(Q_centered, R.T) + centroid_P  # rotate Q and translate it

    return Q_aligned



def rmsd_numpy(P, Q):
    """ Returns the RMSD between structures P and Q """
    Q = kabsch_numpy(P, Q)
    return np.sqrt(np.mean((P - Q) ** 2, axis=(-1, -2)))



def min_rmsd(P, Q_set):
    """
    P: numpy array of size (samples, atoms, 3) representing the output conformations.
    Q_set: numpy array of size (samples, conformations, atoms, 3) representing the sets of possible conformations.
    """
    assert P.shape[0] == Q_set.shape[0] and P.shape[1] == Q_set.shape[2], "Input shapes must match"

    num_samples = P.shape[0]
    num_conformations = Q_set.shape[1]

    rmsd_min = np.zeros(num_samples)
    Q_aligned = np.zeros((num_samples,P.shape[1],3))

    for i in range(num_samples):
        P_i = P[i]
        Q_i_set = Q_set[i]
        rmsd_all = np.zeros(num_conformations)
        Q_i_aligned = np.zeros((num_conformations,P_i.shape[0],3))

        for j in range(num_conformations):
            Q_i_j = Q_i_set[j]
            Q_i_j_aligned = kabsch_numpy_3(P_i[np.newaxis, :, :], Q_i_j[np.newaxis, :, :])
            Q_i_aligned[j,:,:] = Q_i_j_aligned# align Q_i_j to P_i
            rmsd_all[j] = np.sqrt(np.mean((P_i - Q_i_j_aligned) ** 2))  # compute RMSD

        rmsd_min[i] = np.min(rmsd_all)  # store the minimum RMSD for this sample
        Q_aligned[i,:,:] = Q_i_aligned[np.argmin(rmsd_all),:,:]

    return rmsd_min, Q_aligned


import numpy as np
import multiprocessing
from multiprocessing import Pool
from functools import partial


def min_rmsd_parallel(output_conformations, possible_conformations):
    with Pool() as p:
        results = p.starmap(min_rmsd_single_sample, zip(output_conformations, possible_conformations))
        # min_rmsds, Q_aligned = p.starmap(min_rmsd_single_sample, zip(output_conformations, possible_conformations))

    return results


def wrapper_function(output_conformation, possible_conformation):
    return min_rmsd_single_sample(output_conformation, possible_conformation)

def min_rmsd_parallel_2(output_conformations, possible_conformations):
    n_jobs = -1  # Use all available CPUs. Adjust as necessary.
    results = Parallel(n_jobs=n_jobs)(delayed(wrapper_function)(out_conf, poss_conf)
                                      for out_conf, poss_conf in zip(output_conformations, possible_conformations))
    return results

def min_rmsd_single_sample(output_conformation, single_possible_conformations):
    #min_rmsd_val = float('inf')
    #for conformation in single_possible_conformations:
    rmsd_val, q_aligned = min_rmsd(np.expand_dims(output_conformation,0), np.expand_dims(single_possible_conformations,0))

    return [rmsd_val, q_aligned]





def run_main():

    if False:
        num_atoms = 23
        P_x = np.random.rand(1, num_atoms, 3)
        P_a = np.zeros((1,32,3))
        P_a[0,0:23,:] = P_x

        # Apply rotation and translation
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90 degree rotation around z-axis
        translation = np.array([1, 2, 3])
        noise = 0.1 * np.random.rand(1, num_atoms, 3)

        Q_x = np.matmul(P_x, rotation) + translation + noise
        Q   = np.zeros((1,32,3))
        Q[0,0:23,:] = Q_x

        # Apply Kabsch alignment to Q
        Q_aligned = kabsch_numpy_3(P_a, Q)

        print(f"RMSD after Kabsch alignment: {rmsd_numpy(P_a, Q_aligned)}")

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*P_a[0].T, color='blue')
        ax.scatter(*Q[0].T, color='red')
        plt.show()

    # Create samples:
    num_samples = 1000
    P_Set = np.zeros((num_samples, 23, 3))
    num_atoms = 23
    num_conformations = 200

    for xi in range(num_samples):
        # Sample points
        P = np.random.rand(1, num_atoms, 3)
        P_Set[xi, :, :] = P

        # Apply rotation and translation
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90 degree rotation around z-axis
        translation = np.array([1, 2, 3])
        noise = 0.2 * np.random.rand(1, num_atoms, 3)

        Q = np.matmul(P, rotation) + translation + noise

        Q_set = np.zeros((num_samples, num_conformations, num_atoms, 3))

        for zi in range(num_conformations):
            Q_set[xi, zi, :, :] = np.matmul(P, rotation) + translation + (
                        0.075 + (zi % 3) * 0.1 * np.random.rand(1, num_atoms, 3))



    ts_a = time.time()
    #rmsd_a = min_rmsd(P_Set,Q_set)
    # results = min_rmsd_parallel(P_Set,Q_set)
    results = min_rmsd_parallel_2(P_Set, Q_set)
    ts_b = time.time()

    print(f"Time: {ts_b-ts_a}")
    print(f"Original RMSD: {rmsd_numpy( P , Q)}")



class MoleculeDataset_Conformations(Dataset):
    def __init__(self, conformations_file, conformations_vector_file, num_max_conformers=8):
        # Load the data from the .npy files
        self.conformations       = np.load(conformations_file)
        self.conformations_vector = np.load(conformations_vector_file)

    def __len__(self):
        # The length of the dataset is the number of SMILES strings
        return len(self.conformations_vector)

    def __getitem__(self, idx):
        # Get the encoded SMILES string and distance matrix for this index
        conformations = self.conformations[idx]
        conformations_vector = self.conformations_vector[idx]


        # Convert the numpy arrays to PyTorch tensors
        # conformations_flat = torch.tensor(conformations, dtype=torch.float32)
        # conformations_vector_flat = torch.tensor(conformations_vector, dtype=torch.int64)
        # conformations_np = torch.tensor(conformations, dtype=torch.float32)
        # conformations_vector_np = torch.tensor(conformations_vector, dtype=torch.int64)

        return conformations[conformations_vector,:,:]


if __name__ == '__main__':
    cdataset = MoleculeDataset_Conformations("C:/dev/NeonProject/neon/smi64_atoms32_alphabet50_MEDIUM_03_TEST_multipleConformations.npy","C:/dev/NeonProject/neon/smi64_atoms32_alphabet50_MEDIUM_03_TEST_multipleConformationsVector.npy")
    confi = cdataset.__getitem__(123)
    run_main()