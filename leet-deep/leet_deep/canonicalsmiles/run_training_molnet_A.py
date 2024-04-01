import sys
import os
import torch
import deepchem as dc
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
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


# load the molnet data:

# Load Tox21 dataset
tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='Raw')
train_dataset, valid_dataset, test_dataset = datasets

# The feature (i.e., the SMILES string) for each molecule can now be accessed like this:
for X, y, w, id in train_dataset.itersamples():
    print(f'SMILES: {X} Label: {y}')


# data = DataLoader('C:\\dev7\\leet\\smilesdata_025.csv')
# data = DataLoader('C:\\dev7\\leet\\smilesdata_b_05.csv')
# data = DataLoader('C:\\dev7\\leet\\smilesdata_b_1.csv')
data = DataLoader(conf.input_file_data,conf.input_file_alphabet)
data.load()

# try to create output folder
if not os.path.exists(conf.output_dir):
    try:
        os.makedirs(conf.output_dir)
        print("Output folder created successfully.")
    except OSError as e:
        print(f"Error creating output folder: {e}")
else:
    print("Output folder already exists.")


#model = Seq2SeqModel(vocab_size=len(data.alphabet), model_dim=conf.model_dim, num_layers=conf.model_layers) # (A)
model_seq2eq = Seq2SeqModel_2(vocab_size=len(data.alphabet), model_dim=conf.model_dim, num_layers=conf.model_layers) # (B)

## Load model parameters (if provided)

if conf.model_start_file:
    if os.path.isfile(conf.model_start_file):
        # Load the model parameters from the file
        model_seq2eq.load_state_dict(torch.load(conf.model_start_file))
        print("Model parameters loaded successfully.")
    else:
        print("Error: Model start file not found.")
else:
    print("No model start file specified. Model initialized with random parameters.")



# Define the model
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_encoder = model_A.PositionalEncoding(model_dim)
        self.transformer = Transformer(d_model=model_dim, nhead=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)
        x = self.pos_encoder(x)
        x, _, _ = self.transformer(x)
        x = self.output_layer(x[-1, :, :])
        return x


# Define the downstream task model
class DownstreamModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DownstreamModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, output_dim)
        #self.layer2 = nn.Linear(256, 256)
        #self.layer3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Load Tox21 dataset
tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='Raw')
#train_dataset, valid_dataset, test_dataset = datasets

data_loader_molnet = DataLoaderMolNetTox21(data,datasets)
data_loader_molnet.load()
#tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='Raw')
#train_dataset, valid_dataset, test_dataset = datasets

# Convert dataset to tensors
#inputs_train = encodeMolecules([x for x, _, _, _ in train_dataset.itersamples()])
#labels_train = torch.Tensor([y[:1] for _, y, _, _ in train_dataset.itersamples()])
labels_train = torch.Tensor(data_loader_molnet.train_df["Label"].tolist())
inputs_train = torch.Tensor(data_loader_molnet.train_df["Structures"].tolist())

# Create data loaders
train_data = torch.utils.data.TensorDataset(inputs_train, labels_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)  # adjust batch_size as needed

# Instantiate the model
# downstream_model = TransformerClassifier(vocab_size=len(data_loader_molnet.alphabet), model_dim=32, num_layers=3, num_classes=1)  # adjust these parameters as needed

downstream_model = DownstreamModel(conf.model_dim*conf.model_layers*2*40,1)
optimizer = torch.optim.AdamW(downstream_model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()  # for binary classification
# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, verbose=True)



#device ='cuda' #'cpu'
device = conf.device
model_seq2eq     = model_seq2eq.to(device)  # Send the model to the device (CPU or GPU)
downstream_model = downstream_model.to(device)  # Send the model to the device (CPU or GPU)

# Convert DataFrames to PyTorch Datasets
train_dataset = torch.utils.data.TensorDataset(torch.tensor(data_loader_molnet.train_df['Structures'].to_list(), dtype=torch.long),
                                               torch.tensor(data_loader_molnet.train_df['input2'].to_list(), dtype=torch.long),
                                               torch.tensor(data_loader_molnet.train_df['Label'].to_list(), dtype=torch.long))

val_dataset = torch.utils.data.TensorDataset(torch.tensor(data_loader_molnet.val_df['Structures'].to_list(), dtype=torch.long),
                                             torch.tensor(data_loader_molnet.val_df['input2'].to_list(),dtype=torch.long),
                                             torch.tensor(data_loader_molnet.val_df['Label'].to_list(), dtype=torch.long))

# Create DataLoaders
batch_size = conf.optim_batch_size
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(10):
    downstream_model.train()  # probably we need this?
    running_loss = 0.0
    for input1, input2, labels in train_dataloader:
        input1 = input1.to(device)
        input2 = input2.to(device)
        labels = labels.to(torch.long)
        labels = labels.to(device)

        # Use seq2seq model to get the intermediate outputs
        _, inputs_intermediate = model_seq2eq(input1,input2)

        # Flatten the inputs
        #inputs = inputs_d.view(inputs_d.size(0), -1)
        inputs = inputs_intermediate

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = downstream_model(inputs)
        loss = criterion(outputs, labels.view(-1,1).float())
        loss.backward()
        optimizer.step()

        # print statistics
        loss_i = loss.item()
        running_loss += loss_i
    print(f'Epoch: {epoch}, Loss: {running_loss / len(train_loader)}')

    with torch.no_grad():
        loss_v_tot = 0
        count_fully_correct = 0
        all_labels = []
        all_preds = []
        for input1, input2, labels in val_dataloader:
            input1 = input1.to(device)
            input2 = input2.to(device)
            labels = labels.to(torch.long)
            labels = labels.to(device)

            # Use seq2seq model to get the intermediate outputs
            _, inputs_intermediate = model_seq2eq(input1, input2)
            # Flatten the inputs
            # inputs = inputs_d.view(inputs_d.size(0), -1)
            inputs = inputs_intermediate
            outputs = downstream_model(inputs)

            preds_2 = torch.sigmoid(outputs).cpu().numpy() > 0.5
            labels_2 = labels.cpu().numpy()
            all_labels.append(labels_2)
            all_preds.append(preds_2)

            loss = criterion(outputs, labels.view(-1, 1).float())
            loss_i = loss.item()
            loss_v_tot += loss_i
        print(f'Epoch: {epoch}, Validation Loss: {loss_v_tot / len(train_loader)}')

        # Compute metrics
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        roc_auc = roc_auc_score(all_labels, all_preds)

        print(f'Epoch: {epoch}, Validation Loss: {running_loss / len(train_loader)}, Precision: {precision}, Recall: {recall}, F1: {f1}, AUC-ROC: {roc_auc}')


print('Finished Training')