import numpy as np
import torch
import leet_deep.deepspace

from torch.utils.data import DataLoader

from leet_deep.deepspace import NPYDataset
from leet_deep.deepspace import NPYDataset2
from leet_deep.deepspace.model_D_02 import UNet4D_2



def normalize(data, exp, norm_value=100):
    # Calculate the sum of elements in the [][.][.][.] dimensions for each i and j
    if exp:
        data = torch.exp(data)

    sums = data.sum(dim=[1, 2, 3], keepdim=True)  # Shape: [batch_size, 1, 1, 1, 36]
    # Normalize each data[i][..][..][..][j] to the value 100

    mask = (sums < 1)
    cleaned_sums = torch.where(mask, torch.ones_like(sums), sums)

    normalized_data = (data / cleaned_sums) * norm_value
    if (torch.isnan(normalized_data).any()):
        print("NaN!!")


    return normalized_data

def compute_mean(density_grid):
    # Create coordinate grids for each dimension
    x_coords, y_coords, z_coords = torch.meshgrid(
        torch.arange(32), torch.arange(32), torch.arange(32)
    )

    # Flatten the coordinate grids and the density grid
    x_coords_flat = x_coords.reshape(-1)
    y_coords_flat = y_coords.reshape(-1)
    z_coords_flat = z_coords.reshape(-1)
    density_flat = density_grid.reshape(-1)

    # Compute the mean position in 3D space
    mean_x = torch.sum(x_coords_flat * density_flat) / (torch.sum(density_flat)+0.1)
    mean_y = torch.sum(y_coords_flat * density_flat) / (torch.sum(density_flat)+0.1)
    mean_z = torch.sum(z_coords_flat * density_flat) / (torch.sum(density_flat)+0.1)

    return (mean_x,mean_y,mean_z)

def print_mean_similarity(grid_x, grid_target):
    dist_total = 0
    for zi in range(0,36):
        mx = compute_mean(grid_x[zi,:,:,:].squeeze())
        mt = compute_mean(grid_target[:, :, :,zi].squeeze())
        dist_i = np.sqrt( np.sum( (np.array(mx)-np.array(mt)) * (np.array(mx)-np.array(mt)) ) )
        dist_total += dist_i
        print(f"x: {mx} t: {mt} , d= {dist_i}")
    print(f"sum: {dist_total}")


def train(model, dataloader, val_dataloader, optimizer, loss_function, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (x, target, a, b ) in enumerate(dataloader):
            # Move tensors to the right device
            x = x.to(device)
            a = a.to(device)
            b = b.to(device)
            target = target.to(device)

            # Forward pass
            output = model(x.float(), a.float(), b.float())

            if (torch.isnan(output).any()):
                print("NaN!!")

            # Compute the loss
            output_permuted = output.permute(0,2,3,4,1)
            loss = loss_function(normalize(output_permuted,True), normalize(target.float(),False))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update the weights
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}')

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0
        with torch.no_grad():  # No need to compute gradients during validation
            for batch_idx, (val_x, val_target, val_a, val_b) in enumerate(val_dataloader):
                # Move tensors to the right device
                val_x = val_x.to(device)
                val_a = val_a.to(device)
                val_b = val_b.to(device)
                val_target = val_target.to(device)

                # Forward pass
                val_output = model(val_x.float(), val_a.float(), val_b.float())
                val_output_permuted = val_output.permute(0, 2, 3, 4, 1)

                # Compute the validation loss
                val_loss = loss_function(normalize(val_output_permuted,True,norm_value=1000), normalize(val_target.float(),False,norm_value=1000))
                total_val_loss += val_loss.item()

                #if batch_idx % 100 == 0:
                #    print(f'Validation Batch {batch_idx}/{len(val_dataloader)}, Loss: {val_loss.item()}')
                print_mean_similarity(val_output[0,:,:,:,:].squeeze(), val_target[0,:,:,:,:].squeeze())


        # Calculate average validation loss for this epoch
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f'Epoch {epoch}/{num_epochs}, Validation Loss: {avg_val_loss}')



# Prepare the DataLoader, model, optimizer, and loss function

# Specify the file names
filenames = {
    'x': 'C:/temp/deepspace_a/batch_a_x.npy',
    'target': 'C:/temp/deepspace_a/batch_a_target.npy',
    'bondsType': 'C:/temp/deepspace_a/batch_a_bondsType.npy',
    'structureInfo': 'C:/temp/deepspace_a/batch_a_structureInfo.npy'
}
# Create the dataset
dataset = NPYDataset(filenames)

filenames2 = ('C:/temp/deepspace_a/batch_a_0','C:/temp/deepspace_a/batch_a_1','C:/temp/deepspace_a/batch_a_2','C:/temp/deepspace_a/batch_a_3');
dataset2 = NPYDataset2(filenames2)

filenames2val = ('C:/temp/deepspace_a/batch_a_4','C:/temp/deepspace_a/batch_a_5','C:/temp/deepspace_a/batch_a_2','C:/temp/deepspace_a/batch_a_6');
dataset2val = NPYDataset2(filenames2val)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a data loader
batch_size = 16  # Set your batch size
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader = DataLoader(dataset2, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(dataset2val, batch_size=batch_size, shuffle=True)

model = UNet4D_2(36)  # replace `in_channels` and `out_channels` with the correct values
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = torch.nn.MSELoss(reduction='sum')

# Train the model
train(model, dataloader, dataloader_val, optimizer, loss_function, num_epochs=10, device=device)





