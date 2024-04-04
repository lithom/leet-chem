import json
import os
import sys

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn

from leet_deep.canonicalsmiles import GeneralTransformerModel2, GeneralTransformerModel3, GeneralTransformerModel4
from leet_deep.deepspace3 import data_loader_ds3, run_training_ds3_base
from leet_deep.deepspace3.data_loader_ds3 import MoleculeDataset_DS3_A




class ConfigurationHirshfeldData:
    def __init__(self, config_file):
        self.config_file = config_file
        #self.input_file_SmilesEnc = None
        #self.input_file_adjMatrices = None
        #self.input_file_atomCounts = None
        self.input_file = None
        self.output_dir = None
        self.basemodel_dim = None
        self.basemodel_layers = None
        self.basemodel_file = None
        self.optim_learning_rate = None
        self.optim_num_epochs = None
        self.optim_batch_size = None
        self.optim_batches_per_minibatch = None
        self.device = None

    def load_config(self):
        with open(self.config_file) as f:
            config = json.load(f)

        # self.input_file_SmilesEnc = config.get('input_file_smiles')
        # self.input_file_DM = config.get('input_file_dm')
        # self.input_file_cheminfo = config.get('input_file_cheminfo')
        # self.input_file_dist_first = config.get('input_file_dist_first_atom')
        # self.input_file_fullDM = config.get('input_file_full_dm')
        # self.conformer_blinding_rate = config.get('conformer_blinding_rate')
        # self.input_file_conformation = config.get('input_file_conformation')
        self.config = config
        self.input_file = config.get('input_file')
        self.output_dir = config.get('output_dir')
        self.basemodel_dim = config.get('basemodel_dim')
        self.basemodel_layers = config.get('basemodel_layers')
        self.basemodel_file = config.get('basemodel_file')
        self.diffusion_model_dim = config.get('diffusion_model_dim')
        self.diffusion_model_layers = config.get('diffusion_model_layers')
        self.diffusion_model_start_file = config.get('diffusion_model_start_file')
        self.optim_learning_rate = config.get('optim_learning_rate')
        self.optim_num_epochs = config.get('optim_num_epochs')
        self.optim_batch_size = config.get('optim_batch_size')
        self.optim_batches_per_minibatch = config.get('optim_batches_per_minibatch')
        self.device = config.get('device')

    def print_config(self):
        print(f"Output Directory: {self.output_dir}")
        print(f"Model Dimension: {self.basemodel_dim}")
        print(f"Model Layers: {self.basemodel_layers}")
        print(f"Diffusion Model Start File: {self.diffusion_model_start_file}")
        print(f"Optimization Learning Rate: {self.optim_learning_rate}")


def create_dataset(conf: ConfigurationHirshfeldData):
    print('Load data')
    dataset = MoleculeDataset_DS3_A(
        conf.input_file, 'hirshfeld', 8,
        selected_batches=None)

    # Split the Dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create the DataLoaders
    print('Create DataLoaders')
    # batch_size = 256
    batch_size = conf.optim_batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return val_loader, train_loader



def create_model(conf: ConfigurationHirshfeldData):

    base_model = run_training_ds3_base.create_model(conf.basemodel_dim,conf.basemodel_layers)
    print(f'Init Base Model, dim={conf.basemodel_dim}, nlayers={conf.basemodel_layers}') # nhead={basemodel_nhead}, , dim_ff={basemodel_dim_feedforward}')
    # to change for later: dim_expansion_out_1 should then be 8..

    #diffusion_model = GeneralTransformerModel(base_model_full_dim,3,conf.diffusion_model_dim,3,8,conf.diffusion_model_layers)
    diffusion_model = GeneralTransformerModel4(conf.basemodel_dim, 1, length_base_data=32,length_extension_data=64,d_model=conf.diffusion_model_dim, d_out=1, nhead=8, num_decoder_layers=conf.diffusion_model_layers)
    #autoencoder_encoder = AutoEncoderTransformerEncoder(64,32,base_model_full_dim,d_model=8,num_layers=1)
    #autoencoder_decoder = AutoEncoderTransformerDecoder(64,32,base_model_full_dim,d_model=128,num_layers=6)


    ## Load model parameters (if provided)
    if conf.basemodel_file:
        if (os.path.isfile(conf.basemodel_file)):
            # Load the model parameters from the file
            base_model.load_state_dict(torch.load(conf.basemodel_file))
            print("Model parameters loaded successfully.")
        else:
            print("Error: Model start file not found.")
    else:
        print("No model start file specified. Model initialized with random parameters.")


    ## Load model parameters (if provided)
    if conf.diffusion_model_start_file:
        if os.path.isfile(conf.diffusion_model_start_file):
            # Load the model parameters from the file
            diffusion_model.load_state_dict(torch.load(conf.diffusion_model_start_file))
            print("Model parameters loaded successfully.")
        else:
            print("Error: Model start file not found.")
    else:
        print("No model start file specified. Model initialized with random parameters.")



    # Set BaseModel to evaluation mode
    base_model.eval()

    # Ensure we don't update BaseModel's parameters
    for param in base_model.parameters():
        param.requires_grad = False

    # Set autoencoder models to train:
    diffusion_model.train()
    #autoencoder_encoder.train()
    #autoencoder_decoder.train()
    return base_model , diffusion_model



if __name__ == "__main__":
    # Check if the command line argument is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <parameter>")
        sys.exit(1)
    # Retrieve the command line parameter and parse config
    conf_file = sys.argv[1]
    conf = ConfigurationHirshfeldData(conf_file)
    conf.load_config()

    base_model , diffusion_model = create_model(conf)
    val_loader , train_loader = create_dataset(conf)

    print("data loaded")


    # try to create output folder
    if not os.path.exists(conf.output_dir):
        try:
            os.makedirs(conf.output_dir)
            print("Output folder created successfully.")
        except OSError as e:
            print(f"Error creating output folder: {e}")
    else:
        print("Output folder already exists.")



    # Move models to device
    device = conf.device#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = base_model.to(device)
    diffusion_model = diffusion_model.to(device)

    # Optimizer setup
    optimizer = optim.AdamW( diffusion_model.parameters(), lr= conf.optim_learning_rate)

    # Loss function
    criterion = nn.MSELoss()


    # Training loop
    epochs = conf.optim_num_epochs  # Adjust as necessary
    optimizer.zero_grad()  # Step 1: Zero the gradients

    for epoch in range(epochs):
        #batch_cnt = 0
        #for smiles_a, conformations_a , distances_a , num_atoms_a , mask_conformation_flat , mask_distances_flat , diffused_conformer_a , diffused_dm_a , in train_loader:
        for batch_idx, batch in enumerate(train_loader):
            smiles_a, conformations_a , num_atoms_a , hirshfeld , diffused_hirshfeld , mask_atoms_with_h =  batch['smiles_enc'], batch['conformation'], batch['num_atoms'], batch['hirshfeld'], batch['diffused_hirshfeld'], batch['num_atoms_with_hydrogen_mask']

            # ramp up:
            # if(epoch==0):
            #    if(batch_cnt<10):
            #batch_cnt = batch_cnt + 1

            # NOTE!! For the moment we do not use the 3d data of the conformer, this we can still add if needed..

            smiles = smiles_a.to(device)
            conformations = conformations_a.to(device)
            #distances = distances_a.to(device)
            num_atoms = num_atoms_a.to(device)

            atoms_with_h_mask = mask_atoms_with_h.to(device)
            #mask_distances = mask_distances_flat.to(device)
            diffused_hirshfeld = diffused_hirshfeld.to(device)
            diffused_hirshfeld = torch.unsqueeze(diffused_hirshfeld,2)
            diffused_hirshfeld = diffused_hirshfeld * atoms_with_h_mask
            #diffused_dm = diffused_dm_a.to(device)

            # We have to apply this of course to the predicted values..
            hirshfeld = hirshfeld.to(device)
            hirshfeld = torch.unsqueeze(hirshfeld,2)
            mask_atoms_with_h = mask_atoms_with_h.to(device)
            hirshfeld_masked = hirshfeld * mask_atoms_with_h
            #distances_masked = distances * mask_distances
            optimizer.zero_grad()
            # Evaluate BaseModel to get SMILES embeddings
            smiles_embeddings = base_model(smiles,smiles)  # Adjust this call based on BaseModel's actual interface
            # Add noise to conformations for generalization
            #noisy_conformations = add_noise(conformations)
            # Forward pass through the autoencoder encoder part
            predicted_noise_all = diffusion_model(smiles_embeddings[0], diffused_hirshfeld)
            predicted_noise = predicted_noise_all[:,:64,:]
            predicted_noise = predicted_noise * mask_atoms_with_h#conformation_mask
            predicted_noise_scaled = 1.0*predicted_noise
            # Calculate loss
            loss_conf = criterion( diffused_hirshfeld - predicted_noise_scaled, hirshfeld_masked )
            #loss_dist = criterion(reconstructed_distances_scaled, distances_masked)
            loss = loss_conf # + loss_dist # loss_conf + loss_dist
            loss.backward()
            print(f"Loss: {loss.item()}")
            #print(f"Loss_Conf: {loss_conf.item()}  Loss_Dist: {loss_dist.item()}")
            optimizer.step()
        torch.save(diffusion_model.state_dict(), f"{conf.output_dir}/model_{epoch}.pth")

