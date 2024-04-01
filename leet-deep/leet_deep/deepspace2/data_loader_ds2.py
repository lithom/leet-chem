import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class MoleculeDataset_OneHot(Dataset):
    def __init__(self, smiles_file, dist_matrix_file):
        # Load the data from the .npy files
        self.smiles = np.load(smiles_file)
        self.dist_matrices = np.load(dist_matrix_file)

    def __len__(self):
        # The length of the dataset is the number of SMILES strings
        return len(self.smiles)

    def __getitem__(self, idx):
        # Get the encoded SMILES string and distance matrix for this index
        smiles = self.smiles[idx]
        dist_matrix = self.dist_matrices[idx]

        # Apply one-hot encoding to the SMILES string
        smiles_one_hot = np.zeros((smiles.size, 51))  # assuming 51 possible characters (0-50 inclusive)
        smiles_one_hot[np.arange(smiles.size), smiles] = 1

        # Flatten the distance matrix into a 1D tensor
        # and convert the labels into the long type for classification
        dist_matrix_flat = dist_matrix.flatten().astype(np.int64)

        # Convert the numpy arrays to PyTorch tensors
        smiles_one_hot = torch.tensor(smiles_one_hot, dtype=torch.float32)
        dist_matrix_flat = torch.tensor(dist_matrix_flat, dtype=torch.int64)

        return smiles_one_hot, dist_matrix_flat


class MoleculeDataset2(Dataset):
    def __init__(self, config):
        # Load the data from the .npy files
        self.smiles           = np.load(config.config['input_file_smiles'])
        self.adj_matrices     = np.load(config.config['input_file_adj_matrices'])
        self.atom_type_counts = np.load(config.config['input_file_atom_type_matrices'])
        #self.atom_properties = np.load(config.config['input_file_atom_properties'])

    def __len__(self):
        # The length of the dataset is the number of SMILES strings
        return  len(self.smiles)

    def __getitem__(self, idx):
        # Get the encoded SMILES string and distance matrix for this index
        smiles = self.smiles[idx]
        adj_matrices = self.adj_matrices[idx]
        atom_type_counts = self.atom_type_counts[idx]

        smiles_flat       = torch.tensor(smiles, dtype= torch.int64)
        adj_matrices_flat = torch.tensor(adj_matrices, dtype=torch.int64)
        atom_type_counts_flat  = torch.tensor(atom_type_counts, dtype=torch.int64)

        return smiles_flat , adj_matrices_flat, atom_type_counts_flat


class MoleculeDataset3(Dataset):
    def __init__(self, config):
        # Load the data from the .npy files
        self.smiles           = np.load(config.config['input_file_smiles'])
        self.adj_matrices     = np.load(config.config['input_file_adj_matrices'])
        #self.atom_type_counts = np.load(config.config['input_file_atom_type_matrices'])
        self.atom_properties = np.load(config.config['input_file_atom_properties'])
        self.num_atoms       = np.load(config.config['input_file_num_atoms'])

    def __len__(self):
        # The length of the dataset is the number of SMILES strings
        return  len(self.smiles)

    def __getitem__(self, idx):
        # Get the encoded SMILES string and distance matrix for this index
        smiles = self.smiles[idx]
        adj_matrices = self.adj_matrices[idx].transpose(1,2,0)[:,:,:2] # dimension 32x32x2 (cap at 0,1)
        #atom_type_counts = torch.squeeze( self.atom_type_counts[idx] )
        atom_properties = self.atom_properties[idx] # dimension 32x6 (cap maybe at 0..5 ?)
        num_atoms = self.num_atoms[idx]

        smiles_flat       = torch.tensor(smiles, dtype= torch.int64)
        adj_matrices_flat = torch.tensor(adj_matrices, dtype=torch.int64)
        #atom_type_counts_flat  = torch.tensor(atom_type_counts, dtype=torch.int64)
        atom_properties_flat = torch.tensor(atom_properties, dtype=torch.int64)
        num_atoms_flat = torch.tensor(num_atoms, dtype=torch.int64)

        return smiles_flat , adj_matrices_flat, atom_properties_flat , num_atoms_flat #atom_type_counts_flat

def compute_distances(conformation):
    """Computes the pairwise Euclidean distances between atoms in a molecule."""
    distances = np.sqrt(np.sum((conformation[:, np.newaxis, :] - conformation[np.newaxis, :, :]) ** 2, axis=-1))
    return distances

# for simplicity for the moment we just return a single conformer for every structure..
class MoleculeDataset4_3D(Dataset):
    def __init__(self, config, num_diffusion_steps):
        self.smiles = np.load(config.config['input_file_smiles'])
        self.conformers = np.load(config.config['input_file_conformation'])
        self.num_atoms = np.load(config.config['input_file_num_atoms'])
        self.num_diffusion_steps = num_diffusion_steps
        self.beta_schedule = self.beta_schedule = np.linspace(0.001, 2.0, num_diffusion_steps)

    def __len__(self):
        # The length of the dataset is the number of SMILES strings
        return  len(self.smiles)
    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        #conformation = np.squeeze(self.conformers[idx][0,:,:])
        conformation = self.conformers[idx]
        num_atoms_i = self.num_atoms[idx]
        distances_i = compute_distances(conformation)
        mask_distances_i = np.zeros((32, 32))
        mask_distances_i[:num_atoms_i,:num_atoms_i] = 1
        mask_conformation = np.zeros((32,3))
        mask_conformation[:num_atoms_i,:] = 1

        smiles_flat = torch.tensor(smiles, dtype=torch.int64)
        conformation_flat = torch.tensor(conformation, dtype=torch.float32)
        distances_flat = torch.tensor(distances_i, dtype=torch.float32)
        num_atoms_flat = torch.tensor(num_atoms_i, dtype=torch.int64)
        mask_conformation_flat = torch.tensor(mask_conformation, dtype=torch.float32)
        mask_distances_flat = torch.tensor(mask_distances_i, dtype=torch.float32)

        diffused_conformers = [torch.tensor(conformation_flat, dtype=torch.float32)]
        diffused_distances = [torch.tensor(distances_flat, dtype=torch.float32)]

        for beta in self.beta_schedule:
            noise = np.random.normal(0, np.sqrt(beta), conformation.shape)
            diffused_conformer = conformation + noise#np.sqrt(1 - beta) * conformation + noise
            diffused_distance = compute_distances(diffused_conformer)
            diffused_conformers.append(torch.tensor(diffused_conformer, dtype=torch.float32))
            diffused_distances.append(torch.tensor(diffused_distance, dtype=torch.float32))

        # pick a random diffused distance matrix:
        rand_pick = random.randint(0,self.num_diffusion_steps)
        x_diffused_distances = diffused_distances[rand_pick]
        x_diffused_conformer = diffused_conformers[rand_pick]

        return smiles_flat , conformation_flat , distances_flat , num_atoms_flat , mask_conformation_flat , mask_distances_flat , x_diffused_conformer, x_diffused_distances


class MoleculeDataset5_3D(Dataset):
    def __init__(self, path, with3D, num_diffusion_steps, selected_batches=None):
        #super(MoleculeDataset5_3D, self).__init__()
        self.with3D = with3D
        self.smiles_raw_data = []
        self.smiles_data = []
        self.num_atoms_data = []
        self.conformations_data = []
        self.conformations_vector_data = []
        self.adj_matrices_data = []
        self.atom_properties_data = []
        self.num_diffusion_steps = None

        if with3D:
            # Pattern to identify all relevant files
            base_pattern = path#os.path.join(path, "input_file_smiles_b")
            if selected_batches is None:
                # If no specific batches are selected, load all available batches
                data_files = glob.glob(f"{base_pattern}*__Smiles.npy")
                encoded_files = glob.glob(f"{base_pattern}*__SmilesEncoded.npy")
                num_atoms_files = glob.glob(f"{base_pattern}*__numAtoms.npy")
                conformations_files = glob.glob(f"{base_pattern}*__multipleConformations.npy")
                conformations_vector_files = glob.glob(f"{base_pattern}*__multipleConformationsVector.npy")
            else:
                # Load only the specified batches
                data_files = [f"{base_pattern}_batch{batch}___Smiles.npy" for batch in selected_batches]
                encoded_files = [f"{base_pattern}_batch{batch}___SmilesEncoded.npy" for batch in selected_batches]
                num_atoms_files = [f"{base_pattern}_batch{batch}___numAtoms.npy" for batch in selected_batches]
                conformations_files = [f"{base_pattern}_batch{batch}___multipleConformations.npy" for batch in selected_batches]
                conformations_vector_files = [f"{base_pattern}_batch{batch}___multipleConformationsVector.npy" for batch in selected_batches]

            # Load data
            for data_file , enc_file, num_atoms_file , conf_vec_file, conf_file in zip(data_files, encoded_files, num_atoms_files , conformations_vector_files, conformations_files):
                self.smiles_raw_data.append(np.load(data_file))
                self.smiles_data.append(np.load(enc_file))
                self.num_atoms_data.append(np.load(num_atoms_file))
                self.conformations_data.append(np.load(conf_file))
                self.conformations_vector_data.append(np.load(conf_vec_file))

            # Create an index mapping for direct access
            self.index_mapping = []
            for batch_idx, (smiles_enc, conf_vec) in enumerate(zip(self.smiles_data, self.conformations_vector_data)):
                for i, valid_confs in enumerate(conf_vec):
                    for j, is_valid in enumerate(valid_confs):
                        if is_valid:
                            self.index_mapping.append((batch_idx, i, j))

            self.num_diffusion_steps = num_diffusion_steps
            self.beta_schedule = self.beta_schedule = np.linspace(0.001, 2.0, num_diffusion_steps)

        else:
            # Pattern to identify all relevant files
            base_pattern = path
            if selected_batches is None:
                # If no specific batches are selected, load all available batches
                data_files = glob.glob(f"{base_pattern}*__Smiles.npy")
                encoded_files = glob.glob(f"{base_pattern}*__SmilesEncoded.npy")
                num_atoms_files = glob.glob(f"{base_pattern}*__numAtoms.npy")
                adj_matrices_files = glob.glob(f"{base_pattern}*__adjMatrices.npy")
                atom_properties_files = glob.glob(f"{base_pattern}*__atomProperties.npy")
            else:
                # Load only the specified batches
                data_files = [f"{base_pattern}{batch}___Smiles" for batch in selected_batches]
                encoded_files = [f"{base_pattern}{batch}__SmilesEncoded.npy" for batch in selected_batches]
                num_atoms_files = [f"{base_pattern}{batch}__numAtoms.npy" for batch in selected_batches]
                adj_matrices_files = [f"{base_pattern}{batch}__adjMatrices.npy" for batch in selected_batches]
                atom_properties_files = [f"{base_pattern}{batch}__atomProperties.npy" for batch in selected_batches]


            # Load data
            for data_file, enc_file, num_atoms_file, adj_matrices_file , atom_properties_file in zip(data_files, encoded_files,
                                                                                     num_atoms_files, adj_matrices_files,
                                                                                     atom_properties_files):
                self.smiles_raw_data.append(np.load(data_file))
                self.smiles_data.append(np.load(enc_file))
                self.num_atoms_data.append(np.load(num_atoms_file))
                self.adj_matrices_data.append(np.load(adj_matrices_file))
                self.atom_properties_data.append(np.load(atom_properties_file))

            # Create an index mapping for direct access
            self.index_mapping = []
            for batch_idx, smiles_enc in enumerate(self.smiles_data):
                for i, smiles_i in enumerate(smiles_enc):
                    self.index_mapping.append((batch_idx, i))




    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        if self.with3D:
            batch_idx, smile_idx, conf_idx = self.index_mapping[idx]
            smile_raw = self.smiles_raw_data[batch_idx][smile_idx]
            smile_enc = self.smiles_data[batch_idx][smile_idx]
            num_atoms = self.num_atoms_data[batch_idx][smile_idx]
            conf_data = self.conformations_data[batch_idx][smile_idx, conf_idx]
            conf_data_tensor = torch.tensor(conf_data, dtype=torch.float32)

            mask_conformation = np.zeros((32, 3))
            mask_conformation[:num_atoms, :] = 1

            # diffused conformers:
            diffused_conformers = [torch.tensor(conf_data_tensor, dtype=torch.float32)]
            for beta in self.beta_schedule:
                noise = np.random.normal(0, np.sqrt(beta), conf_data.shape)
                diffused_conformer = conf_data + noise#np.sqrt(1 - beta) * conformation + noise
                diffused_conformers.append(torch.tensor(diffused_conformer, dtype=torch.float32))

            # pick a random diffused conformer:
            rand_pick = random.randint(0, self.num_diffusion_steps)
            x_diffused_conformer = diffused_conformers[rand_pick]

            return {
                'smiles': smile_raw,
                'smiles_enc': torch.tensor(smile_enc, dtype=torch.int64),
                'num_atoms': torch.tensor(num_atoms, dtype=torch.int64),
                'conformation': conf_data_tensor,
                'mask_conformation': mask_conformation,
                'diffused_conformation': x_diffused_conformer
            }
        else:
            batch_idx, smile_idx = self.index_mapping[idx]
            smile_raw = self.smiles_raw_data[batch_idx][smile_idx]
            smile_enc = self.smiles_data[batch_idx][smile_idx]
            num_atoms = self.num_atoms_data[batch_idx][smile_idx]
            adj_matrices = self.adj_matrices_data[batch_idx][smile_idx].transpose(1,2,0)[:,:,:2] # only take two, and permute
            atom_properties = self.atom_properties_data[batch_idx][smile_idx]

            return {
                'smile': smile_raw,
                'smile_enc': torch.tensor(smile_enc, dtype=torch.int64),
                'num_atoms': torch.tensor(num_atoms, dtype=torch.int64),
                'adj_matrices': torch.tensor(adj_matrices, dtype=torch.int64),
                'atom_properties': torch.tensor(atom_properties, dtype=torch.int64)
            }



class MoleculeDataset_Int(Dataset):
    def __init__(self, smiles_file, dist_matrix_file, cheminfo_file, dist_first_atom_file, full_dist_matrix_file, conformations_file,
                 conformer_bounding_box_length, conformer_blinding_rate):
        # Load the data from the .npy files
        self.smiles = np.load(smiles_file)
        self.dist_matrices = np.load(dist_matrix_file)
        self.cheminfo = np.load(cheminfo_file)
        self.dist_first_atom = np.load(dist_first_atom_file)
        self.full_dist_matrices = np.load(full_dist_matrix_file)
        self.conformations = np.load(conformations_file)
        self.conformer_bounding_box_length = conformer_bounding_box_length
        self.conformer_blinding_rate = conformer_blinding_rate

        self.ri = random.Random()
        #self.ri.seed(123)

    def __len__(self):
        # The length of the dataset is the number of SMILES strings
        return  len(self.smiles)

    def __getitem__(self, idx):
        # Get the encoded SMILES string and distance matrix for this index
        smiles = self.smiles[idx]
        dist_matrix = self.dist_matrices[idx]
        dist_first_atom = self.dist_first_atom[idx]

        # Apply one-hot encoding to the SMILES string
        smiles_int = smiles
        #smiles_one_hot[np.arange(smiles.size), smiles] = 1

        # Flatten the distance matrix into a 1D tensor
        # and convert the labels into the long type for classification
        dist_matrix_flat = dist_matrix.flatten().astype(np.int64)

        cheminfo_matrix = self.cheminfo[idx]

        full_dist_matrix = self.full_dist_matrices[idx]
        conformation_matrix_full = self.conformations[idx]

        # scale conformation matrix into -0.5 , 0.5:
        # NOTE: here we divide by 4 * the halfwidth of the box
        # BUT:  the output of the transformer will be in -1..1, so to scale back we multiply only by 2 * the halfwidth
        conformation_matrix_full = conformation_matrix_full / (4*self.conformer_bounding_box_length)

        # compute blinding for conformation:
        blinding_vector = np.zeros((full_dist_matrix.shape[0]))
        num_atoms = np.sum( full_dist_matrix[0,:] != 0 )
        for xi in range(num_atoms):
            rv = self.ri.random()
            if rv < self.conformer_blinding_rate:
                blinding_vector[xi] = 1

        conformation_matrix_blinded = conformation_matrix_full.copy()
        conformation_matrix_blinded[:,blinding_vector==1] = 0


        # Convert the numpy arrays to PyTorch tensors
        smiles_int = torch.tensor(smiles_int, dtype=torch.int64)
        dist_matrix_flat = torch.tensor(dist_matrix_flat, dtype=torch.int64)
        cheminfo_matrix_flat = torch.tensor(cheminfo_matrix, dtype=torch.int64)
        dist_first_atom_flat = torch.tensor(dist_first_atom, dtype=torch.int64)
        conformation_matrix_blinded_flat = torch.tensor(conformation_matrix_blinded, dtype=torch.float32)
        conformation_matrix_flat = torch.tensor(conformation_matrix_full, dtype=torch.float32)
        full_dist_matrix_flat = torch.tensor(full_dist_matrix, dtype=torch.int64)
        conformation_blinding_mask = torch.tensor(blinding_vector, dtype=torch.int64)

        return smiles_int, dist_matrix_flat, cheminfo_matrix_flat, dist_first_atom_flat, full_dist_matrix_flat, conformation_matrix_blinded_flat, conformation_matrix_flat, conformation_blinding_mask

# # Create the Dataset
# dataset = MoleculeDataset('smiles.npy', 'dist_matrices.npy')
#
# # Split the Dataset into training and validation sets
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
#
# # Create the DataLoaders
# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)

#xa = np.load('C:\dev7\leet\smi64_atoms32_alphabet50_MEDIUM_FIRST_3D_2_conformation.npy')
#print('mkay')