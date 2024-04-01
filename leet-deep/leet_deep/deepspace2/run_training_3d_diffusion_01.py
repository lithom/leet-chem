import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from leet_deep.canonicalsmiles import Seq2SeqModel_5, AutoEncoderTransformerEncoder, AutoEncoderTransformerDecoder, \
    GeneralTransformerModel
from leet_deep.canonicalsmiles.leet_transformer import CustomTransformerEncoder
from leet_deep.deepspace2.data_loader_ds2 import MoleculeDataset4_3D, MoleculeDataset5_3D


class ConfigurationLocalData:
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
        self.autoencoder_encoder_start_file = None
        self.autoencoder_decoder_start_file = None
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
        print(f"Diffusion Model Start File: {self.autoencoder_encoder_start_file}")
        print(f"Optimization Learning Rate: {self.optim_learning_rate}")


def create_dataset(conf : ConfigurationLocalData ):
    # Create the Dataset
    print(f'Init Dataset..')
    #dataset = MoleculeDataset4_3D(conf,8) # this works, but we want to replace this..
    #dataset = MoleculeDataset5_3D(conf.input_file,True, 8,selected_batches=(1,2,3,4))
    #dataset = MoleculeDataset5_3D(conf.input_file, True, 8, selected_batches=None)
    dataset = MoleculeDataset5_3D(conf.input_file, True, 8, selected_batches=(1,2,3,4,5,6,7,8,9,10))

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



def create_model(conf_file):

    conf = ConfigurationLocalData(conf_file)
    conf.load_config()

    # try to create output folder
    if not os.path.exists(conf.output_dir):
        try:
            os.makedirs(conf.output_dir)
            print("Output folder created successfully.")
        except OSError as e:
            print(f"Error creating output folder: {e}")
    else:
        print("Output folder already exists.")

    # Hyperparameters Base Model
    basemodel_n_tokens = 64 # sequence length
    basemodel_nhead = 8
    basemodel_dim_feedforward = 2048 #1024

    #warmup_steps = 200
    #warmup_steps = 6
    val_loader, train_loader = create_dataset(conf)

    print(f'Init Model, dim={conf.basemodel_dim}, nhead={basemodel_nhead}, nlayers={conf.basemodel_layers}, dim_ff={basemodel_dim_feedforward}')
    # to change for later: dim_expansion_out_1 should then be 8..

    expansions = []
    # ( num_classes, (shape_of_elements_out) )
    # total_elements_out must be multiple of sequence length 64
    # shape_of_elements_out MUST INCLUDE classes, in fact last element of shape_of_elements MUST BE num classes
    #expansions.append( (2,(32,32,8,2)) ) # adj matrix
    expansions.append( (2,(32,32,2,2)) ) # adj matrix
    expansions.append( (2,(32,6,8)) ) # chem properties (0..7) for the counts.
    #expansions.append( (16,(32,13,4,16)) ) # atom counts

    base_model = Seq2SeqModel_5(sequence_length=64,vocab_size_in=44,expansions_list=expansions,model_dim=conf.basemodel_dim, num_layers=conf.basemodel_layers) # (B)

    base_model_full_dim = conf.basemodel_dim * 2 * conf.basemodel_layers # this is the size of the full output (all layers, x2 because encoder / decoders layers)

    diffusion_model = GeneralTransformerModel(base_model_full_dim,3,conf.diffusion_model_dim,3,8,conf.diffusion_model_layers)
    #autoencoder_encoder = AutoEncoderTransformerEncoder(64,32,base_model_full_dim,d_model=8,num_layers=1)
    #autoencoder_decoder = AutoEncoderTransformerDecoder(64,32,base_model_full_dim,d_model=128,num_layers=6)


    ## Load model parameters (if provided)
    if conf.basemodel_file:
        if os.path.isfile(conf.basemodel_file):
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
    return base_model , diffusion_model , conf


if __name__ == "__main__":

    # Check if the command line argument is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <parameter>")
        sys.exit(1)
    # Retrieve the command line parameter and parse config
    conf_file = sys.argv[1]
    base_model , diffusion_model , conf = create_model(conf_file)
    val_loader , train_loader = create_dataset(conf)


    # Move models to device
    device = conf.device#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = base_model.to(device)
    diffusion_model = diffusion_model.to(device)
    #autoencoder_encoder = autoencoder_encoder.to(device)
    #autoencoder_decoder = autoencoder_decoder.to(device)


    #def add_noise(data, mean=0.0, std=0.01):
    #    return data + torch.randn_like(data) * std

    # Optimizer setup for the autoencoder
    optimizer = optim.AdamW( diffusion_model.parameters(), lr= conf.optim_learning_rate)

    # Loss function
    criterion = nn.MSELoss()

    def generate_conformation_mask(num_atoms, max_atoms=32):
        """Generates a mask for conformations based on the actual number of atoms."""
        batch_size = num_atoms.size(0)
        mask = torch.arange(max_atoms).expand(batch_size, max_atoms).to(num_atoms.device)
        mask = (mask < num_atoms[:, None]).float()
        return mask

    def generate_distance_mask(num_atoms, max_atoms=32):
        """Generates a mask for distance matrices based on the actual number of atoms."""
        conformation_mask = generate_conformation_mask(num_atoms, max_atoms)
        distance_mask = conformation_mask[:, :, None] * conformation_mask[:, None, :]
        return distance_mask


    # Training loop
    epochs = conf.optim_num_epochs  # Adjust as necessary
    for epoch in range(epochs):
        batch_cnt = 0
        #for smiles_a, conformations_a , distances_a , num_atoms_a , mask_conformation_flat , mask_distances_flat , diffused_conformer_a , diffused_dm_a , in train_loader:
        for batch in train_loader:
            smiles_a, conformations_a , num_atoms_a , mask_conformation_flat , diffused_conformer_a =  batch['smiles_enc'], batch['conformation'], batch['num_atoms'], batch['mask_conformation'], batch['diffused_conformation']

            # ramp up:
            # if(epoch==0):
            #    if(batch_cnt<10):
            batch_cnt = batch_cnt + 1
            smiles = smiles_a.to(device)
            conformations = conformations_a.to(device)
            #distances = distances_a.to(device)
            num_atoms = num_atoms_a.to(device)
            mask_conformation = mask_conformation_flat.to(device)
            #mask_distances = mask_distances_flat.to(device)
            diffused_conformer = diffused_conformer_a.to(device)
            #diffused_dm = diffused_dm_a.to(device)

            # Generate masks
            #conformation_mask = generate_conformation_mask(num_atoms).unsqueeze(-1)  # Add an extra dimension for broadcasting
            #distance_mask = generate_distance_mask(num_atoms)
            # We have to apply this of course to the predicted values..
            conformations_masked = conformations * mask_conformation
            #distances_masked = distances * mask_distances
            optimizer.zero_grad()
            # Evaluate BaseModel to get SMILES embeddings
            smiles_embeddings = base_model(smiles,smiles)  # Adjust this call based on BaseModel's actual interface
            # Add noise to conformations for generalization
            #noisy_conformations = add_noise(conformations)
            # Forward pass through the autoencoder encoder part
            predicted_noise_all = diffusion_model(smiles_embeddings[0], diffused_conformer)
            predicted_noise = predicted_noise_all[:,:32,:]
            predicted_noise = predicted_noise * mask_conformation#conformation_mask
            predicted_noise_scaled = predicted_noise * 25.0
            # Calculate loss
            loss_conf = criterion( diffused_conformer - predicted_noise_scaled, conformations_masked)
            #loss_dist = criterion(reconstructed_distances_scaled, distances_masked)
            loss = loss_conf # + loss_dist # loss_conf + loss_dist
            loss.backward()
            print(f"Loss: {loss.item()}")
            #print(f"Loss_Conf: {loss_conf.item()}  Loss_Dist: {loss_dist.item()}")
            optimizer.step()

        # Validate..
        diffusion_model.eval()
        total_loss_adj = 0
        total_loss_cp = 0
        correct_preds = 0
        total_values_correct = 0
        total_values = 0
        batch_cnt = 0
        with torch.no_grad():
            ts_eval_a = time.time()
            #for smiles_a, conformations_a, distances_a, num_atoms_a, mask_conformation_flat, mask_distances_flat, diffused_conformer_a, diffused_dm_a, in val_loader:
            for batch in val_loader:
                smiles_a, conformations_a, num_atoms_a, mask_conformation_flat, diffused_conformer_a = batch['smiles_enc'], batch['conformation'], batch['num_atoms'], batch['mask_conformation'], batch['diffused_conformation']
                # ramp up:
                # if(epoch==0):
                #    if(batch_cnt<10):
                batch_cnt = batch_cnt + 1
                smiles = smiles_a.to(device)
                conformations = conformations_a.to(device)
                #distances = distances_a.to(device)
                num_atoms = num_atoms_a.to(device)
                mask_conformation = mask_conformation_flat.to(device)
                #mask_distances = mask_distances_flat.to(device)
                diffused_conformer = diffused_conformer_a.to(device)
                #diffused_dm = diffused_dm_a.to(device)
                # Generate masks
                # conformation_mask = generate_conformation_mask(num_atoms).unsqueeze(-1)  # Add an extra dimension for broadcasting
                # distance_mask = generate_distance_mask(num_atoms)
                # We have to apply this of course to the predicted values..
                conformations_masked = conformations * mask_conformation
                # distances_masked = distances * mask_distances
                #optimizer.zero_grad()
                # Evaluate BaseModel to get SMILES embeddings
                smiles_embeddings = base_model(smiles, smiles)  # Adjust this call based on BaseModel's actual interface
                # Add noise to conformations for generalization
                # noisy_conformations = add_noise(conformations)
                # Forward pass through the autoencoder encoder part
                predicted_noise_all = diffusion_model(smiles_embeddings[0], diffused_conformer)
                predicted_noise = predicted_noise_all[:, :32, :]
                predicted_noise = predicted_noise * mask_conformation  # conformation_mask
                predicted_noise_scaled = predicted_noise * 25.0
                # Calculate loss
                loss_conf = criterion(diffused_conformer - predicted_noise_scaled, conformations_masked)
                # loss_dist = criterion(reconstructed_distances_scaled, distances_masked)
                loss = loss_conf  # + loss_dist # loss_conf + loss_dist
                #loss.backward()
                print(f"Validation Loss: {loss.item()}")
                # print(f"Loss_Conf: {loss_conf.item()}  Loss_Dist: {loss_dist.item()}")
                #optimizer.step()

        print(f"Epoch {epoch+1}")
        print(f"forward: sec per sample: {(time.time() - ts_eval_a) / len(val_loader.dataset)}")
        # save models:
        torch.save(diffusion_model.state_dict(), f"{conf.output_dir}/diffusion_model{epoch}.pth")
        #torch.save(autoencoder_decoder.state_dict(), f"{conf.output_dir}/model_decoder_{epoch}.pth")

