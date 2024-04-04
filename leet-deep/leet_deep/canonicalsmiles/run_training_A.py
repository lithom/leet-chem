import sys
import os
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

# Define a custom loss function to ignore the 'y' characters
def custom_loss(y_pred, y_true):
    mask = y_true.ne(data.c2i('y'))  # Create mask where y_true is not 'y'
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    # Flatten the output tensor from (batch_size, sequence_length, vocab_size) to (batch_size*sequence_length, vocab_size)
    y_pred_masked = y_pred_masked.view(-1, y_pred_masked.size(-1))
    # Flatten the target tensor from (batch_size, sequence_length) to (batch_size*sequence_length)
    y_true_masked = y_true_masked.view(-1)
    return nn.CrossEntropyLoss()(y_pred_masked, y_true_masked)


#model = Seq2SeqModel(vocab_size=len(data.alphabet), model_dim=conf.model_dim, num_layers=conf.model_layers) # (A)
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


#device ='cuda' #'cpu'
device = conf.device
model = model.to(device)  # Send the model to the device (CPU or GPU)

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=conf.optim_learning_rate, weight_decay=0.01)

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, verbose=True)

# Convert DataFrames to PyTorch Datasets
train_dataset = torch.utils.data.TensorDataset(torch.tensor(data.train_df['input1'].to_list(), dtype=torch.long),
                                               torch.tensor(data.train_df['input2'].to_list(), dtype=torch.long),
                                               torch.tensor(data.train_df['output'].to_list(), dtype=torch.long))

val_dataset = torch.utils.data.TensorDataset(torch.tensor(data.val_df['input1'].to_list(), dtype=torch.long),
                                             torch.tensor(data.val_df['input2'].to_list(), dtype=torch.long),
                                             torch.tensor(data.val_df['output'].to_list(), dtype=torch.long))

# Create DataLoaders
batch_size = conf.optim_batch_size
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = conf.optim_num_epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input1, input2, output in train_dataloader:
        input1 = input1.to(device)
        input2 = input2.to(device)
        output = output.to(device)
        optimizer.zero_grad()
        y_pred = model(input1, input2)
        loss = custom_loss(y_pred, output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    print(f"Epoch: {epoch}, Loss: {total_loss / len(train_dataloader)}")

    torch.save(model.state_dict(), f"{conf.output_dir}/model_{epoch}.pth")
    # Validation
    model.eval()
    with torch.no_grad():
        with open( f"{conf.output_dir}/epoch_{epoch}.txt", "w") as f:
            loss_v_tot = 0
            count_fully_correct = 0
            for input1, input2, target in val_dataloader:
                input1 = input1.to(device)
                input2 = input2.to(device)
                target = target.to(device)
                output_predictions = model(input1, input2)
                prediction = torch.argmax(output_predictions, dim=-1)
                loss_v = custom_loss(output_predictions, target)
                loss_v_tot += loss_v.item()

                for inp1, inp2, pred, tar, cert , cert_sum in zip(input1, input2, prediction, target, torch.max(output_predictions, dim=-1)[0] , torch.sum( torch.exp(output_predictions), dim=(2))):
                    inp1 = inp1.cpu().numpy()
                    inp2 = inp2.cpu().numpy()
                    pred = pred.cpu().numpy()
                    tar = tar.cpu().numpy()
                    cert = cert.cpu().numpy()
                    cert_sum = cert_sum.cpu().numpy()
                    cert_p = np.exp(cert) / cert_sum
                    bitmask1 = np.where(pred == tar, 1, 0)
                    bitmask2 = np.where(inp2 == data.c2i('y'), 1, 0)
                    bitmask  = bitmask1 | bitmask2
                    count_fully_correct += (1 if bitmask.all() else 0)

                    grades = []
                    for c in cert_p:
                        if c > 0.9:
                            grade = 'E'
                        elif c > 0.80:
                            grade = 'D'
                        elif c > 0.60:
                            grade = 'C'
                        elif c > 0.50:
                            grade = 'B'
                        else:
                            grade = 'A'
                        grades.append(grade)

                    # Convert integer sequences back to characters
                    inp1_string = "".join([data.i2c(i) for i in inp1])
                    pred_string = "".join([data.i2c(i) for i in pred])
                    tar_string = "".join([data.i2c(i) for i in tar])
                    bitmask_string = "".join([str(i) for i in bitmask])
                    grades_string = "".join(grades)

                    f.write(f"{inp1_string} {tar_string} {pred_string} {bitmask_string} {grades_string}\n")
            print(f"Epoch: {epoch}, LossValidation: {loss_v_tot / len(val_dataloader)}")
            print(f"Epoch: {epoch}, Validation: fully correct:  {count_fully_correct} / {len(val_dataset)}")
            with open(f"{conf.output_dir}/log.csv", "a") as f2:
                f2.write(f"{epoch},{total_loss / len(train_dataloader)},{loss_v_tot / len(val_dataloader)},{count_fully_correct},{(1.0*count_fully_correct)/len(val_dataset)}\n")


