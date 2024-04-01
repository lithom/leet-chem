from leet_deep.deepspace2 import MoleculeTransformer
from leet_deep.deepspace2 import MoleculeDataset_Int
from leet_deep.canonicalsmiles import Seq2SeqModel_4
from leet_deep.deepspace2 import Molecule3DTransformer
from leet_deep.deepspace2 import Molecule3DTransformer2
import torch
import time
import sys
import json
import os
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch import nn


class ConfigurationDM:
    def __init__(self, config_file):
        self.config_file = config_file
        self.input_file_SmilesEnc = None
        self.input_file_DM        = None
        self.input_file_cheminfo  = None
        self.input_file_dist_first = None
        self.input_file_fullDM = None
        self.input_file_conformation = None
        self.conformer_bounding_box_length = None
        self.conformer_blinding_rate = None
        self.output_dir = None
        self.model_A_dim = None
        self.model_A_layers = None
        self.model_A_start_file = None
        self.model_3d_dim = None
        self.model_3d_layers = None
        self.model_3d_start_file = None
        self.optim_learning_rate = None
        self.optim_num_epochs = None
        self.optim_batch_size = None
        self.optim_batches_per_minibatch = None
        self.device = None

    def load_config(self):
        with open(self.config_file) as f:
            config = json.load(f)

        self.input_file_SmilesEnc = config.get('input_file_smiles')
        self.input_file_DM = config.get('input_file_dm')
        self.input_file_cheminfo = config.get('input_file_cheminfo')
        self.input_file_dist_first = config.get('input_file_dist_first_atom')
        self.input_file_fullDM = config.get('input_file_full_dm')
        self.conformer_bounding_box_length = config.get('conformer_bounding_box_length')
        self.conformer_blinding_rate = config.get('conformer_blinding_rate')
        self.input_file_conformation = config.get('input_file_conformation')
        self.output_dir = config.get('output_dir')
        self.model_A_dim = config.get('model_A_dim')
        self.model_A_layers = config.get('model_A_layers')
        self.model_A_start_file = config.get('model_A_start_file')
        self.model_3d_dim = config.get('model_3d_dim')
        self.model_3d_layers = config.get('model_3d_layers')
        self.model_3d_start_file = config.get('model_3d_start_file')
        self.optim_learning_rate = config.get('optim_learning_rate')
        self.optim_num_epochs = config.get('optim_num_epochs')
        self.optim_batch_size = config.get('optim_batch_size')
        self.optim_batches_per_minibatch = config.get('optim_batches_per_minibatch')
        self.device = config.get('device')

    def print_config(self):
        print(f"Input Files: {self.input_file_DM} , {self.input_file_SmilesEnc}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Model Dimension: {self.model_dim}")
        print(f"Model Layers: {self.model_layers}")
        print(f"Model Start File: {self.model_start_file}")
        print(f"Optimization Learning Rate: {self.optim_learning_rate}")


def create_dataset(conf):
    # Create the Dataset
    print(f'Init Dataset: {conf.input_file_SmilesEnc} / {conf.input_file_DM}')
    dataset = MoleculeDataset_Int(conf.input_file_SmilesEnc, conf.input_file_DM, conf.input_file_cheminfo,
                                  conf.input_file_dist_first,conf.input_file_fullDM,conf.input_file_conformation,conf.conformer_bounding_box_length,conf.conformer_blinding_rate)  # MoleculeDataset_Int('C:/dev7/leet/smi64_atoms32_alphabet50_SmilesEncoded.npy', 'C:/dev7/leet/smi64_atoms32_alphabet50_DM.npy')

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


# Check if the command line argument is provided
if len(sys.argv) < 2:
    print("Usage: python script.py <parameter>")
    sys.exit(1)

# Retrieve the command line parameter and parse config
conf_file = sys.argv[1]
conf = ConfigurationDM(conf_file)
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


device = conf.device#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
n_tokens = 64 # sequence length
nhead = 8
dim_feedforward = 2048 #1024
lr = conf.optim_learning_rate
#warmup_steps = 200
#warmup_steps = 6
num_epochs = conf.optim_num_epochs

val_loader, train_loader = create_dataset(conf)

## Create the model A
print(f'Init Model, dim={conf.model_A_dim}, nhead={nhead}, nlayers={conf.model_A_layers}, dim_ff={dim_feedforward}')
# to change for later: dim_expansion_out_1 should then be 8..


expansions = []
expansions.append( (17,32) ) # distance matrix
model_A = Seq2SeqModel_4(vocab_size_in=45,expansions_list=expansions,model_dim=conf.model_A_dim, num_layers=conf.model_A_layers) # (B)

# Now construct the Molecule3DTransformer:
#model_3d = Molecule3DTransformer(45,conf.model_3d_dim,64,64*32*6)
model_3d = Molecule3DTransformer2(45)



## Load model parameters (if provided)
if conf.model_A_start_file:
    if os.path.isfile(conf.model_A_start_file):
        # Load the model parameters from the file
        model_A.load_state_dict(torch.load(conf.model_A_start_file))
        print("Model (A) parameters loaded successfully.")
    else:
        print("Error: Model (A) start file not found.")
else:
    print("No model_A start file specified. Model initialized with random parameters.")

# Freeze parameters of the pretrained model so they don't get updated during training
for param in model_A.parameters():
    param.requires_grad = False
model_A.eval()  # Set the model to evaluation mode

## Load 3d model parameters (if provided)
if conf.model_3d_start_file:
    if os.path.isfile(conf.model_3d_start_file):
        # Load the model parameters from the file
        model_3d.load_state_dict(torch.load(conf.model_3d_start_file))
        print("Model (3d) parameters loaded successfully.")
    else:
        print("Error: Model (3d) start file not found.")
else:
    print("No model_3d start file specified. Model initialized with random parameters.")



def train_loop(dataloader, model_A, model_3d, optimizer, criterion, device, num_batches_per_minibatch):
    model_3d.train()
    total_loss = 0

    # Reset gradients
    optimizer.zero_grad()

    for idx, (smiles_data, dmf, chimf, dfaf, fdmf, coords_blinded, coords_full, blinding) in enumerate(dataloader):
        smiles_data = smiles_data.to(device)
        coords_blinded = coords_blinded.to(device)
        coords_full    = coords_full.to(device)
        blinding = blinding.to(device)

        # Generate additional data with the smiles_model
        model_A.eval()
        with torch.no_grad():
            additional_data = model_A(smiles_data,smiles_data)

        # Predict and compute loss
        output = model_3d(smiles_data, additional_data[1], additional_data[0], coords_blinded, blinding)
        loss = criterion(100*output, 100*coords_full)

        # back propagation
        model_A.train() # chatgpt says we should do this (?)
        loss.backward()

        #if (idx+1) % num_batches_per_minibatch == 0:
        optimizer.step()
        model_3d.train()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_loop(dataloader, model_A, model_3d, optimizer, criterion, device):
    model_A.eval()  # set the model to eval mode
    model_3d.eval()

    ts_train_a = time.time()
    sum_all_distances = 0
    sum_max_distances = 0
    total_loss = 0
    with torch.no_grad():  # no need to calculate gradients
        for idx, (smiles_data, dmf, chimf, dfaf, fdmf, coords_blinded, coords_full, blinding) in enumerate(dataloader):
            smiles_data = smiles_data.to(device)
            coords_blinded = coords_blinded.to(device)
            coords_full = coords_full.to(device)
            blinding = blinding.to(device)

            # Generate additional data with the smiles_model
            additional_data = model_A(smiles_data, smiles_data)

            # Predict and compute loss
            output = model_3d(smiles_data, additional_data[1], additional_data[0], coords_blinded, blinding)
            loss = criterion(100*output, 100*coords_full)

            # Compute the Euclidean distance between all 3D coordinates
            # Reshape the tensors to (sample, 32, 3) to calculate distances along the 32 3D coordinates
            output_reshaped = 100*output.transpose(1, 2).contiguous().view(-1, 32, 3)
            target_reshaped = 100*coords_full.transpose(1, 2).contiguous().view(-1, 32, 3)

            # Calculate the squared Euclidean distance element-wise
            distance_sq = torch.sum((output_reshaped - target_reshaped) ** 2, dim=-1)
            sum_all_distances += torch.sum( distance_sq[:] ).item()

            # Get the maximum Euclidean distance for each sample along the 32 3D coordinates
            max_distance       = torch.max(distance_sq, dim=-1).values
            sum_max_distances += torch.sum( max_distance[:] ).item()

            total_loss += loss.item()

        print(f"forward: ms per sample: {(time.time() - ts_train_a) / len(dataloader.dataset)}")
        print(f"sum all distances: {(sum_all_distances) / len(dataloader.dataset)}, sum max distances: {(sum_max_distances) / len(dataloader.dataset)}")
        return total_loss / len(dataloader)
    return total_loss / len(dataloader)


model_A  = model_A.to(device)
model_3d = model_3d.to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model_3d.parameters(), lr=conf.optim_learning_rate, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, verbose=True)

# run training:

for epoch in range(num_epochs):
    loss_train = train_loop(train_loader,model_A,model_3d,optimizer,criterion,device,conf.optim_batches_per_minibatch)
    loss_val   = validate_loop(val_loader,model_A,model_3d,optimizer,criterion,device)
    scheduler.step()

    print(
        f'Epoch: {epoch + 1}, Train Loss: {loss_train:.4f}, Val Loss: {loss_val:.4f}')
    torch.save(model_3d.state_dict(), f"{conf.output_dir}/model3d_{epoch}.pth")




