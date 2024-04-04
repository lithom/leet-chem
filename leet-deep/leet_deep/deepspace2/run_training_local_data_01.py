from leet_deep.deepspace2 import MoleculeTransformer, MoleculeDataset3
from leet_deep.deepspace2 import MoleculeDataset_Int
from leet_deep.deepspace2 import MoleculeDataset2
from leet_deep.canonicalsmiles import Seq2SeqModel_5
import torch
import time
import sys
import json
import os
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch import nn

from leet_deep.deepspace2.data_loader_ds2 import MoleculeDataset5_3D


class ConfigurationLocalData:
    def __init__(self, config_file):
        self.config_file = config_file
        #self.input_file_SmilesEnc = None
        #self.input_file_adjMatrices = None
        #self.input_file_atomCounts = None
        self.input_file = None

        self.output_dir = None
        self.model_dim = None
        self.model_layers = None
        self.model_start_file = None
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
        self.model_dim = config.get('model_dim')
        self.model_layers = config.get('model_layers')
        self.model_start_file = config.get('model_start_file')
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


def create_dataset(conf : ConfigurationLocalData ):
    # Create the Dataset
    print(f'Init Dataset..')
    #dataset = MoleculeDataset2(conf)  # MoleculeDataset_Int('C:/dev7/leet/smi64_atoms32_alphabet50_SmilesEncoded.npy', 'C:/dev7/leet/smi64_atoms32_alphabet50_DM.npy')
    #dataset = MoleculeDataset3(conf) # THIS ONE WORKS, BUT WE NOW SWITCH TO THE NEW FORMAT..
    dataset = MoleculeDataset5_3D(conf.input_file,False,8,None)

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


def train(model, dataloader, optimizer, criterion_adj, criterion_cp, device, num_batches_per_minibatch):
    model.train()
    total_loss_adj = 0
    total_loss_cp = 0
    batchcnt = 0

    ts_train_a = time.time()

    # Reset the gradients
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        # Move data to the target device

        #inputs, targets_adj_matrix, targets_chem_prop = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        inputs, targets_adj_matrix, targets_chem_prop = batch['smile_enc'].to(device), batch['adj_matrices'].to(device), batch['atom_properties'].to(device)

        # Perform forward pass
        # outputs, outputs_2 = model(inputs)
        outputs_all = model(inputs, inputs)

        output_adj_matrix = outputs_all[1]
        output_chem_prop  = outputs_all[2]
        batchcnt = batchcnt + 1

        # permute..
        output_adj_matrix_permuted = output_adj_matrix.permute(0,4,1,2,3)
        output_chem_prop_permuted = output_chem_prop.permute(0,3,1,2)

        targets_adj_matrix_permuted = targets_adj_matrix.permute(0,1,2,3)
        #targets_atom_types_permuted = targets_atom_types.permute(0,1,3,2)
        targets_chem_prop_permuted = targets_chem_prop.permute(0,1,2)

        # clip
        #output_adj_matrix_permuted_clipped  = output_adj_matrix_permuted[:,:,(0,8),:,: ]
        #targets_adj_matrix_permuted_clipped = targets_adj_matrix_permuted[:,(0,8),:,:]

        #output_atom_types_permuted_clipped  = output_atom_types_permuted[:,:,(0,8),:,: ]
        #targets_atom_types_permuted_clipped = targets_atom_types_permuted[:,(0,8),:, :]
        # no clipping needed..
        output_adj_matrix_permuted_clipped = output_adj_matrix_permuted
        targets_adj_matrix_permuted_clipped = targets_adj_matrix_permuted
        output_chem_prop_permuted_clipped = output_chem_prop_permuted
        targets_chem_prop_permuted_clipped = targets_chem_prop_permuted

        #loss_adj_matrix  = criterion_adj(output_adj_matrix_permuted_clipped, targets_adj_matrix_permuted_clipped)
        #loss_atom_types  = criterion_at(output_atom_types_permuted_clipped, targets_atom_types_permuted_clipped )
        loss_adj_matrix  = criterion_adj(output_adj_matrix_permuted_clipped, targets_adj_matrix_permuted_clipped)
        loss_chem_prop   = criterion_cp(output_chem_prop_permuted_clipped, targets_chem_prop_permuted_clipped)
        #loss_atom_types  = criterion_at(output_atom_types_permuted_clipped, targets_atom_types_permuted_clipped )

        loss_total = loss_adj_matrix + loss_chem_prop #+ loss_atom_types
        # Perform backward pass and optimization
        loss_total.backward()

        if (i+1) % num_batches_per_minibatch == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update total loss
        total_loss_adj += loss_adj_matrix.item()
        total_loss_cp  += loss_chem_prop.item()
        #total_loss_at  += loss_atom_types.item()

    print(f"time training: {time.time()-ts_train_a}")
    print(f"loss_adj: {total_loss_adj} , loss_cp: {total_loss_cp}")
    return ( (loss_total) / len(dataloader) , (total_loss_adj) / len(dataloader) , (total_loss_cp) / len(dataloader) )

def validate(model, dataloader, criterion_adj, criterion_cp, device):
    model.eval()
    total_loss_adj = 0
    total_loss_cp  = 0
    correct_preds = 0
    total_values_correct = 0
    total_values = 0

    with torch.no_grad():
        ts_eval_a = time.time()
        batchcnt = 0
        for batch in dataloader:
            # Move data to the target device
            #inputs, targets_adj_matrix, targets_chem_prop = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            inputs, targets_adj_matrix, targets_chem_prop = batch['smile_enc'].to(device), batch['adj_matrices'].to(device), batch['atom_properties'].to(device)

            # Perform forward pass
            # outputs, outputs_2 = model(inputs)
            outputs_all = model(inputs, inputs)

            output_adj_matrix = outputs_all[1]
            #output_atom_types = outputs_all[2]
            output_chem_prop = outputs_all[2]
            batchcnt = batchcnt + 1

            # permute..
            output_adj_matrix_permuted = output_adj_matrix.permute(0, 4, 1, 2, 3)
            output_chem_prop_permuted = output_chem_prop.permute(0, 3, 1, 2)

            targets_adj_matrix_permuted = targets_adj_matrix.permute(0, 1, 2, 3)
            # targets_atom_types_permuted = targets_atom_types.permute(0,1,3,2)
            targets_chem_prop_permuted = targets_chem_prop.permute(0, 1, 2)

            # clip
            # output_adj_matrix_permuted_clipped  = output_adj_matrix_permuted[:,:,(0,8),:,: ]
            # targets_adj_matrix_permuted_clipped = targets_adj_matrix_permuted[:,(0,8),:,:]

            # output_atom_types_permuted_clipped  = output_atom_types_permuted[:,:,(0,8),:,: ]
            # targets_atom_types_permuted_clipped = targets_atom_types_permuted[:,(0,8),:, :]
            # no clipping needed..
            output_adj_matrix_permuted_clipped = output_adj_matrix_permuted
            targets_adj_matrix_permuted_clipped = targets_adj_matrix_permuted
            output_chem_prop_permuted_clipped = output_chem_prop_permuted
            targets_chem_prop_permuted_clipped = targets_chem_prop_permuted

            # loss_adj_matrix  = criterion_adj(output_adj_matrix_permuted_clipped, targets_adj_matrix_permuted_clipped)
            # loss_atom_types  = criterion_at(output_atom_types_permuted_clipped, targets_atom_types_permuted_clipped )
            loss_adj_matrix = criterion_adj(output_adj_matrix_permuted_clipped, targets_adj_matrix_permuted_clipped)
            loss_chem_prop = criterion_cp(output_chem_prop_permuted_clipped, targets_chem_prop_permuted_clipped)
            # loss_atom_types  = criterion_at(output_atom_types_permuted_clipped, targets_atom_types_permuted_clipped )

            loss_total = loss_adj_matrix + loss_chem_prop

            total_loss_adj += loss_adj_matrix.item()
            total_loss_cp  += loss_chem_prop.item()

            # # Get the predicted classes (indices of maximum values along the last dimension)
            # predicted_classes = torch.argmax(output_dm_reshaped, dim=1)
            # # Compare predicted classes with target classes to check for correctness
            # correct = (predicted_classes == targets_full_matrix_reshaped)
            # # Count the number of correct samples for all classes
            # num_correct_samples = correct.all(dim=-1).sum()
            # correct_preds += num_correct_samples.item()
            # total_values += len(predicted_classes.flatten())
            # total_values_correct += sum(correct.flatten()).item()

        print(f"forward: sec per sample: { (time.time()-ts_eval_a)/len(dataloader.dataset)}")
        #print(f"fully correct: {correct_preds} / {len(dataloader.dataset)}")
        #print(f"correct values: {total_values_correct} / {total_values}")
    return ( (loss_total) / len(dataloader) , (total_loss_adj) / len(dataloader) , (total_loss_cp) / len(dataloader) ,  correct_preds)



# Check if the command line argument is provided
if len(sys.argv) < 2:
    print("Usage: python script.py <parameter>")
    sys.exit(1)

# Retrieve the command line parameter and parse config
conf_file = sys.argv[1]
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

## Create the model, loss function, optimizer, and scheduler
print(f'Init Model, dim={conf.model_dim}, nhead={nhead}, nlayers={conf.model_layers}, dim_ff={dim_feedforward}')
# to change for later: dim_expansion_out_1 should then be 8..

expansions = []
# ( num_classes, (shape_of_elements_out) )
# total_elements_out must be multiple of sequence length 64
# shape_of_elements_out MUST INCLUDE classes, in fact last element of shape_of_elements MUST BE num classes
#expansions.append( (2,(32,32,8,2)) ) # adj matrix
expansions.append( (2,(32,32,2,2)) ) # adj matrix
expansions.append( (2,(32,6,8)) ) # chem properties (0..7) for the counts.
#expansions.append( (16,(32,13,4,16)) ) # atom counts

model = Seq2SeqModel_5(sequence_length=64,vocab_size_in=44,expansions_list=expansions,model_dim=conf.model_dim, num_layers=conf.model_layers) # (B)

## Load model parameters (if provided)
if conf.model_start_file:
    if os.path.isfile(conf.model_start_file):
        # Load the model parameters from the file
        model.load_state_dict(torch.load(conf.model_start_file))
        print("Model parameters loaded successfully.")
    else:
        print("Error: Model start file not found.")
else:
    print("No model start file specified. Model initialized with random parameters.")


# class weights:
#class_weights_adj_matrix = torch.tensor( [ 1 , 32*32 / 8 ] ).to(device)
criterion_adj = nn.CrossEntropyLoss(weight=None)
criterion_at  = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=conf.optim_learning_rate, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, verbose=True)
model = model.to(device)


#adjacency_matrix_parts = [0,4,8,12]


# Training loop
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion_adj, criterion_at, device, conf.optim_batches_per_minibatch)
    val_loss = validate(model, val_loader, criterion_adj,criterion_at, device)
    scheduler.step()

    print(
        f'Epoch: {epoch + 1}, Train Loss Adj: {train_loss[1]:.4f}, Train Loss AT: {train_loss[2]:.4f}, Val Loss Adj: {val_loss[1]:.4f}, Val Loss AT: {val_loss[2]}')
    torch.save(model.state_dict(), f"{conf.output_dir}/model_{epoch}.pth")
    # torch.save(model.state_dict(), f"model_{epoch}.pth")