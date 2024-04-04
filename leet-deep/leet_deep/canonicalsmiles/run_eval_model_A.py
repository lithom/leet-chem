import sys
import os
import time
from leet_deep.canonicalsmiles import *


import json

class Configuration:
    def __init__(self, config_file):
        self.config_file = config_file
        self.input_file_data     = None
        self.input_file_alphabet = None
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

        self.input_file_data = config.get('input_file_data')
        self.input_file_alphabet = config.get('input_file_alphabet')
        self.output_dir = config.get('output_dir')
        self.model_dim = config.get('model_dim')
        self.model_layers = config.get('model_layers')
        self.model_start_file = config.get('model_start_file')
        self.optim_learning_rate = config.get('optim_learning_rate')
        self.optim_num_epochs = config.get('optim_num_epochs')
        self.optim_batch_size = config.get('optim_batch_size')
        self.device = config.get('device')

    def print_config(self):
        print(f"Input File: {self.input_file}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Model Dimension: {self.model_dim}")
        print(f"Model Layers: {self.model_layers}")
        print(f"Model Start File: {self.model_start_file}")
        print(f"Optimization Learning Rate: {self.optim_learning_rate}")






# Check if the command line argument is provided
if len(sys.argv) < 2:
    print("Usage: python script.py <parameter>")
    sys.exit(1)

# Retrieve the command line parameter and parse config
conf_file = sys.argv[1]
conf = Configuration(conf_file)
conf.load_config()


data = DataLoader(conf.input_file_data,conf.input_file_alphabet)
data.load()

model = Seq2SeqModel_2(vocab_size=len(data.alphabet), model_dim=conf.model_dim, num_layers=conf.model_layers) # (B)


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



device = conf.device
model = model.to(device)  # Send the model to the device (CPU or GPU)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(data.train_df['input1'].to_list(), dtype=torch.long),
                                               torch.tensor(data.train_df['input2'].to_list(), dtype=torch.long),
                                               torch.tensor(data.train_df['output'].to_list(), dtype=torch.long))


# Create DataLoaders
batch_size = conf.optim_batch_size
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Eval loop
#num_epochs = conf.optim_num_epochs

model.eval()
total_loss = 0
print("GO!")
ts_a = time.time()
for input1, input2, output in train_dataloader:
    input1 = input1.to(device)
    input2 = input2.to(device)
    output = output.to(device)
    #optimizer.zero_grad()
    y_pred = model(input1, input2)
    #loss = custom_loss(y_pred, output)
    #loss.backward()
    #optimizer.step()
    #total_loss += loss.item()
#scheduler.step()

ts_b = time.time()
print("Done!")
print( f"time: {ts_b-ts_a} , { (1.0*ts_b-ts_a) / len(data.train_df) }" )
print( "mkay..")



