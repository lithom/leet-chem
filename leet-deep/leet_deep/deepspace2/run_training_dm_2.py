import leet_deep
from leet_deep.deepspace2 import MoleculeTransformer
from leet_deep.deepspace2 import MoleculeDataset_Int
from leet_deep.canonicalsmiles import Seq2SeqModel_2
from leet_deep.canonicalsmiles import Seq2SeqModel_3
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
        self.output_dir = None
        self.model_dim = None
        self.model_layers = None
        self.model_start_file = None
        self.optim_learning_rate = None
        self.optim_num_epochs = None
        self.optim_batch_size = None
        self.device = None

    def load_config(self):
        with open(self.config_file) as f:
            config = json.load(f)

        self.input_file_SmilesEnc = config.get('input_file_smiles')
        self.input_file_DM = config.get('input_file_dm')
        self.input_file_cheminfo = config.get('input_file_cheminfo')
        self.input_file_dist_first = config.get('input_file_dist_first_atom')
        self.input_file_full_DM = config.get('input_file_full_dm')
        self.input_file_conformation = config.get('input_file_conformation')
        self.conformer_blinding_rate = config.get('conformer_blinding_rate')
        self.output_dir = config.get('output_dir')
        self.model_dim = config.get('model_dim')
        self.model_layers = config.get('model_layers')
        self.model_start_file = config.get('model_start_file')
        self.optim_learning_rate = config.get('optim_learning_rate')
        self.optim_num_epochs = config.get('optim_num_epochs')
        self.optim_batch_size = config.get('optim_batch_size')
        self.device = config.get('device')

    def print_config(self):
        print(f"Input Files: {self.input_file_DM} , {self.input_file_SmilesEnc}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Model Dimension: {self.model_dim}")
        print(f"Model Layers: {self.model_layers}")
        print(f"Model Start File: {self.model_start_file}")
        print(f"Optimization Learning Rate: {self.optim_learning_rate}")






# def collapse_classes(tensor, old_class_range, new_class):
#     tensor = tensor.clone()  # Create a copy to avoid changing the original tensor
#     summed_logits = tensor.narrow(-1, old_class_range[0], old_class_range[1] - old_class_range[0] + 1).sum(-1)
#     tensor = tensor.narrow(-1, 0, new_class)  # Keep the elements before the new class
#     tensor = torch.cat((tensor, summed_logits.unsqueeze(-1)), dim=-1)  # Insert the summed logits
#     return tensor


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    ts_train_a = time.time()
    for batch in dataloader:
        # Move data to the target device
        inputs, targets_matrix, targets_cheminfo, targets_dist_first = batch[0].to(device), batch[1].to(device), batch[
            2].to(device), batch[3].to(device)

        # Reset the gradients
        optimizer.zero_grad()

        # Perform forward pass
        # outputs, outputs_2 = model(inputs)
        outputs_all = model(inputs, inputs)

        outputs_1 = outputs_all[0]
        outputs_2 = outputs_all[1]

        # Compute loss
        # outputs = outputs.permute(0,2,1)
        outputs_1 = outputs_1.permute(0, 2, 1)
        outputs_2 = outputs_2.permute(0, 2, 1)
        #print(f'target min: {targets.min()}, target max: {targets.max()}')
        #print(f'output shape: {outputs.shape}, output type: {outputs.dtype}')
        #print(f'target shape: {targets.shape}, target type: {targets.dtype}')
        #loss = criterion(outputs.view(batch[0].size()[0],-1, 16), targets.view(batch[0].size()[0],-1))

        #outputs = torch.clamp(outputs, min=0, max=4)  # cap the prediction at 8
        #targets = torch.clamp(targets, min=0, max=4)  # cap the targets at 4
        #targets_2 = torch.clamp(targets_2, min=0, max=4)  # cap the targets at 4
        #outputs = collapse_classes(outputs,(4,15),4)
        # targets = torch.clamp(targets, min=0, max=8)  # cap the targets at 8
        # targets_a = targets.view(targets.size()[0],32,32);
        # targets_a = targets_a[:,0:2,0:16]
        # outputs_a = outputs.view(outputs.size()[0],outputs.size()[1],2,32);
        # outputs_a = outputs_a[:,:,0:2,0:16]

        #loss = criterion(outputs, targets)
        #loss = criterion(outputs_1, targets_cheminfo) + criterion(outputs_2[:,:,0:32], targets_dist_first) #criterion(outputs,targets) + 4 * criterion(outputs_2, targets_2)
        loss = criterion(outputs_2[:, :, 0:32], targets_dist_first)  # criterion(outputs,targets) + 4 * criterion(outputs_2, targets_2)


        # Perform backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update total loss
        total_loss += loss.item()

    print(f"time training: {time.time()-ts_train_a}")
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss_1 = 0
    total_loss_2 = 0
    correct_preds = 0

    with torch.no_grad():
        ts_eval_a = time.time()
        batchcnt = 0
        for batch in dataloader:
            inputs, targets_matrix, targets_cheminfo, targets_dist_first = batch[0].to(device), batch[1].to(device), batch[2].to(device) , batch[3].to(device)

            #outputs, outputs_2 = model(inputs)
            outputs_all  = model(inputs,inputs)

            outputs_1 = outputs_all[0]
            outputs_2 = outputs_all[1]

            #outputs = torch.clamp(outputs, min=0, max=8)  # cap the prediction at 8
            #targets = torch.clamp(targets, min=0, max=8)  # cap the targets at 8
            #outputs = collapse_classes(outputs, (4, 15), 4)
            #targets   = torch.clamp(targets, min=0, max=4)  # cap the targets at 4
            #targets_2 = torch.clamp(targets_2, min=0, max=4)  # cap the targets at 4

            # Convert output probabilities to predicted class.
            #_, preds = torch.max(outputs, dim=2)

            #if batchcnt==0:
                #leet_deep.deepspace2.outputhelper.output_results(f"out_data_{epoch}.txt",outputs,targets.view(-1,32,32)[:,0:2,:],2,32)
            batchcnt = batchcnt+1

            # targets = torch.clamp(targets, min=0, max=8)  # cap the targets at 8
            # targets_a = targets.view(targets.size()[0], 32, 32);
            # targets_a = targets_a[:, 0:2, 0:32]

            # Count the number of fully correct matrices.
            #correct_preds += torch.sum((preds == targets).all(dim=(1)).float())
            #outputs = outputs.permute(0, 2, 1)
            outputs_1 = outputs_1.permute(0, 2, 1)
            outputs_2 = outputs_2.permute(0, 2, 1)

            #outputs_a = outputs.view(outputs.size()[0], outputs.size()[1], 2, 32);
            #outputs_a = outputs_a[:, :, 0:2, 0:32]

            #loss_1 = criterion(outputs_1, targets_cheminfo)
            #loss_2 = 4 * criterion(outputs_2, targets_2)
            loss_2 = criterion(outputs_2[:,:,0:32], targets_dist_first)
            # validation: print(outputs_2[0,:,0:16]) print(targets_dist_first[0,0:16])
            # loss = criterion(outputs_a.flatten(2), targets_a.flatten(1))
            #loss = criterion(outputs.view(-1, 16), targets.view(-1))

            #total_loss_1 += loss_1.item()
            total_loss_2 += loss_2.item()

        print(f"forward: ms per sample: { (time.time()-ts_eval_a)/len(dataloader.dataset)}")
    return ( (total_loss_1+total_loss_2) / len(dataloader) , correct_preds)






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
#d_model = 51
n_tokens = 64 # sequence length
#d_model = 128
#d_model = 256#1024
#d_model = conf.model_dim
#nhead = 8
nhead = 8
#num_layers = 8
#num_layers = 4
#num_layers = conf.model_layers
#dim_feedforward = 2048
dim_feedforward = 2048 #1024
#lr = 0.0001
lr = conf.optim_learning_rate
#warmup_steps = 5000
#warmup_steps = 6
num_epochs = conf.optim_num_epochs




# Create the model, loss function, optimizer, and scheduler
print(f'Init Model, dim={conf.model_dim}, nhead={nhead}, nlayers={conf.model_layers}, dim_ff={dim_feedforward}')
#model = MoleculeTransformer(64,8,32,8,conf.model_dim, nhead, conf.model_layers, dim_feedforward).to(device)
# to change for later: dim_expansion_out_1 should then be 8..
model = Seq2SeqModel_3(vocab_size_in=45,vocab_size_out_1=5,vocab_size_out_2=6,dim_expansion_out_1=1,model_dim=conf.model_dim, num_layers=conf.model_layers) # (B)



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

#model.load_state_dict(torch.load('model_32.pth'))


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=conf.optim_learning_rate, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, verbose=True)
model = model.to(device)

# Implement learning rate warmup
# def adjust_lr(optimizer, step_num, warmup_steps=warmup_steps):
#     lr = d_model**-0.5 * min(step_num**-0.5, step_num * warmup_steps**-1.5)
#     print(f"adjust lr: {lr}")
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

# Training loop
for epoch in range(num_epochs):
    #adjust_lr(optimizer, epoch+1)
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device, epoch)
    scheduler.step()

    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss[0]:.4f}, Fully correct: {val_loss[1]} / {len(val_loader.dataset)}')
    torch.save(model.state_dict(), f"{conf.output_dir}/model_{epoch}.pth")
    #torch.save(model.state_dict(), f"model_{epoch}.pth")