import glob
import os

import torch
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


# mode can be either: 'base' , '3d' , 'hirshfeld'
class MoleculeDataset_DS3_A(Dataset):
    def __init__(self, path, mode, num_diffusion_steps, selected_batches=None):
        #super(MoleculeDataset5_3D, self).__init__()
        self.mode = mode
        self.with3D = (mode=='3d' or mode=='hirshfeld')
        self.smiles_raw_data = []
        self.smiles_data = []
        self.num_atoms_data = []
        self.conformations_data = []
        self.conformations_vector_data = []
        self.hirshfeld_data = []
        self.num_atoms_with_hydrogen_data = []
        self.adj_matrices_data = []
        self.atom_properties_data = []
        self.atom_properties_2_data = []
        self.dist_matrix_data = []
        self.plane_rule_constraints_data = []
        self.num_diffusion_steps = None

        if self.with3D:
            # Pattern to identify all relevant files
            base_pattern = path#os.path.join(path, "input_file_smiles_b")
            if selected_batches is None:
                # If no specific batches are selected, load all available batches
                data_files = glob.glob(f"{base_pattern}*__Smiles.npy")
                encoded_files = glob.glob(f"{base_pattern}*__SmilesEncoded.npy")
                num_atoms_files = glob.glob(f"{base_pattern}*__numAtoms.npy")
                conformations_files = glob.glob(f"{base_pattern}*__multipleConformations.npy")
                conformations_vector_files = glob.glob(f"{base_pattern}*__multipleConformationsVector.npy")
                hirshfeld_files = glob.glob(f"{base_pattern}*___hirshfeldCharges.npy")
            else:
                # Load only the specified batches
                data_files = [f"{base_pattern}_batch{batch}___Smiles.npy" for batch in selected_batches]
                encoded_files = [f"{base_pattern}_batch{batch}___SmilesEncoded.npy" for batch in selected_batches]
                num_atoms_files = [f"{base_pattern}_batch{batch}___numAtoms.npy" for batch in selected_batches]
                conformations_files = [f"{base_pattern}_batch{batch}___multipleConformations.npy" for batch in selected_batches]
                conformations_vector_files = [f"{base_pattern}_batch{batch}___multipleConformationsVector.npy" for batch in selected_batches]
                hirshfeld_files = [f"{base_pattern}_batch{batch}___hirshfeldCharges.npy" for batch in selected_batches]

            # Load data
            for data_file , enc_file, num_atoms_file , conf_vec_file, conf_file , hirshfeld_file in zip(data_files, encoded_files, num_atoms_files , conformations_vector_files, conformations_files , hirshfeld_files):
                smiles_raw_data_i = np.load(data_file)
                self.smiles_raw_data.append(smiles_raw_data_i)
                self.smiles_data.append(np.load(enc_file))
                self.num_atoms_data.append(np.load(num_atoms_file))
                self.conformations_data.append(np.load(conf_file))
                self.conformations_vector_data.append(np.load(conf_vec_file))
                self.hirshfeld_data.append(np.load(hirshfeld_file))
                # use rdkit to get all atoms (indluding h)

                num_hydrogen_data_i = np.zeros((len(smiles_raw_data_i),1),dtype=np.int64)
                for idx_i , sxi in enumerate(smiles_raw_data_i):
                    mol = Chem.MolFromSmiles(sxi.decode().replace('y',''))
                    mol = Chem.AddHs(mol)
                    num_atoms = mol.GetNumAtoms()
                    num_hydrogen_data_i[idx_i]=num_atoms
                self.num_atoms_with_hydrogen_data.append(num_hydrogen_data_i)
                #check that sizes agree..
                #if added_i != len(smiles_raw_data_i):
                #    raise Exception('Error, number of added smiles disagrees..')


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
                atom_properties2_files = glob.glob(f"{base_pattern}*__atomProperties2.npy")
                dist_matrix_files = glob.glob(f"{base_pattern}*__fullDM.npy")
                plane_rule_constraints_files = glob.glob(f"{base_pattern}*__sharedPlaneRule.npy")
            else:
                # Load only the specified batches
                data_files = [f"{base_pattern}_batch{batch}___Smiles.npy" for batch in selected_batches]
                encoded_files = [f"{base_pattern}_batch{batch}___SmilesEncoded.npy" for batch in selected_batches]
                num_atoms_files = [f"{base_pattern}_batch{batch}___numAtoms.npy" for batch in selected_batches]
                adj_matrices_files = [f"{base_pattern}_batch{batch}___adjMatrices.npy" for batch in selected_batches]
                atom_properties_files = [f"{base_pattern}_batch{batch}___atomProperties.npy" for batch in selected_batches]
                atom_properties2_files = [f"{base_pattern}_batch{batch}___atomProperties2.npy"  for batch in selected_batches]
                dist_matrix_files = [f"{base_pattern}_batch{batch}___fullDM.npy" for batch in selected_batches]
                plane_rule_constraints_files = [f"{base_pattern}_batch{batch}___sharedPlaneRule.npy" for batch in selected_batches]



            # Load data
            for data_file, enc_file, num_atoms_file, adj_matrices_file , atom_properties_file , atom_properties2_file ,\
                    dist_matrix_file, plane_rule_constraints_file in zip(data_files, encoded_files,
                                                                                     num_atoms_files, adj_matrices_files,
                                                                                     atom_properties_files , atom_properties2_files,
                                                                                     dist_matrix_files, plane_rule_constraints_files):
                self.smiles_raw_data.append(np.load(data_file))
                self.smiles_data.append(np.load(enc_file))
                self.num_atoms_data.append(np.load(num_atoms_file))
                self.adj_matrices_data.append(np.load(adj_matrices_file))
                self.atom_properties_data.append(np.load(atom_properties_file))
                self.atom_properties_2_data.append(np.load(atom_properties2_file))
                self.dist_matrix_data.append(np.load(dist_matrix_file))
                self.plane_rule_constraints_data.append(np.load(plane_rule_constraints_file))


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

            if(self.mode=='3d'):
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
            if (self.mode == 'hirshfeld'):
                hirshfeld_data = self.hirshfeld_data[batch_idx][smile_idx]
                diffused_hirshfeld_data = []
                # diffused conformers:
                num_diffusion_steps_hirshfeld = 100
                hirshfeld_beta_schedule = np.linspace(0.005,1.0,100) #1.0*np.logspace(0.01, 4.0, num_diffusion_steps_hirshfeld)
                for beta in hirshfeld_beta_schedule:
                    beta_a = beta
                    noise = np.random.normal(0, np.sqrt( beta_a ), hirshfeld_data.shape)
                    diffused_hx = hirshfeld_data + noise  # np.sqrt(1 - beta) * conformation + noise
                    diffused_hirshfeld_data.append(torch.tensor(diffused_hx, dtype=torch.float32))

                # pick a random diffused conformer:
                rand_pick = random.randint(0, num_diffusion_steps_hirshfeld-1)
                x_diffused_hirshfeld = diffused_hirshfeld_data[rand_pick]

                num_atoms_with_hydrogen_mask = torch.zeros(64,1)
                num_atoms_with_hydrogen_mask[:int(self.num_atoms_with_hydrogen_data[batch_idx][smile_idx]),0] = 1.0

                return {
                    'smiles': smile_raw,
                    'smiles_enc': torch.tensor(smile_enc, dtype=torch.int64),
                    'num_atoms': torch.tensor(num_atoms, dtype=torch.int64),
                    'conformation': conf_data_tensor,
                    'mask_conformation': mask_conformation,
                    'hirshfeld':hirshfeld_data,
                    'diffused_hirshfeld': x_diffused_hirshfeld,
                    'num_atoms_with_hydrogen': self.num_atoms_with_hydrogen_data[batch_idx][smile_idx],
                    'num_atoms_with_hydrogen_mask': num_atoms_with_hydrogen_mask
                }


        else:
            batch_idx, smile_idx = self.index_mapping[idx]
            smile_raw = self.smiles_raw_data[batch_idx][smile_idx]
            smile_enc = self.smiles_data[batch_idx][smile_idx]
            num_atoms = self.num_atoms_data[batch_idx][smile_idx]
            adj_matrices = self.adj_matrices_data[batch_idx][smile_idx].transpose(1,2,0)[:,:,:2] # only take two, and permute
            atom_properties = self.atom_properties_data[batch_idx][smile_idx]
            atom_properties2 = self.atom_properties_2_data[batch_idx][smile_idx]
            distance_matrices = self.dist_matrix_data[batch_idx][smile_idx]
            plane_rule_constraints = self.plane_rule_constraints_data[batch_idx][smile_idx]

            left_upper_mask = torch.zeros((32,32),dtype=torch.float32)
            left_upper_mask[:num_atoms,:num_atoms] = 1.0

            return {
                'smile': smile_raw,
                'smile_enc': torch.tensor(smile_enc, dtype=torch.int64),
                'num_atoms': torch.tensor(num_atoms, dtype=torch.int64),
                'adj_matrices': torch.tensor(adj_matrices, dtype=torch.int64),
                'distance_matrices': torch.tensor(distance_matrices, dtype=torch.float32),
                'atom_properties': torch.tensor(atom_properties, dtype=torch.int64),
                'atom_properties2': torch.tensor(atom_properties2, dtype=torch.int64),
                'plane_rule_constraints': torch.tensor(plane_rule_constraints, dtype=torch.int64),
                'left_upper_mask': left_upper_mask
            }