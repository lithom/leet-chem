
from leet_deep.canonicalsmiles import *


# data = DataLoader('C:\\dev7\\leet\\smilesdata_025.csv')
data = DataLoader('C:\\dev7\\leet\\smilesdata_b_05.csv')
data.load()

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


model = Seq2SeqModel(vocab_size=len(data.alphabet), model_dim=128, num_layers=4)
#model = Transformer2(input_dim=len(data.alphabet), model_dim=512, output_dim=len(data.alphabet), num_layers=6)

device ='cuda' #'cpu'
model = model.to(device)  # Send the model to the device (CPU or GPU)


# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

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
batch_size = 256
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = 400
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

    torch.save(model.state_dict(), f"model_{epoch}.pth")
    # Validation
    model.eval()
    with torch.no_grad():
        with open(f"epoch_{epoch}.txt", "w") as f:
            for input1, input2, target in val_dataloader:
                input1 = input1.to(device)
                input2 = input2.to(device)
                output_predictions = model(input1, input2)
                prediction = torch.argmax(output_predictions, dim=-1)

                for pred, tar, cert , cert_sum in zip(prediction, target, torch.max(output_predictions, dim=-1)[0] , torch.sum( torch.exp(output_predictions), dim=(2))):
                    pred = pred.cpu().numpy()
                    tar = tar.cpu().numpy()
                    cert = cert.cpu().numpy()
                    cert_sum = cert_sum.cpu().numpy()
                    cert_p = np.exp(cert) / cert_sum
                    bitmask = np.where(pred == tar, 0, 1)

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
                    pred_string = "".join([data.i2c(i) for i in pred])
                    tar_string = "".join([data.i2c(i) for i in tar])
                    bitmask_string = "".join([str(i) for i in bitmask])
                    grades_string = "".join(grades)

                    f.write(f"{tar_string} {pred_string} {bitmask_string} {grades_string}\n")

