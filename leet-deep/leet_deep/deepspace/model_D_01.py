import torch
import torch.nn as nn

def conv3d_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet4D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encoder = nn.ModuleList([
            nn.Sequential(
                #conv3d_block(in_channels, 64),
                conv3d_block(in_channels, in_channels),
                #nn.MaxPool3d((2, 2, 2, 1))
                nn.MaxPool3d((2, 2, 2))
            ),
            nn.Sequential(
                conv3d_block(in_channels, 128),
                #nn.MaxPool3d((2, 2, 2, 1))
                nn.MaxPool3d((2, 2, 2))
            ),
            # Add more layers if needed
        ])

        self.decoder = nn.ModuleList([
            nn.Sequential(
                #nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2, 1)),
                nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2)),
                conv3d_block(128, 64)
            ),
            # Add more layers if needed
        ])

        self.pointwise_conv = nn.Conv3d(2*36, in_channels, 1)

        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

        # # Define MLPs for the extra inputs
        # self.mlp_a = nn.Sequential(
        #     nn.Linear(36 * 9, 64),
        #     nn.ReLU(inplace=True),
        #     # Add more layers if needed
        # )
        # self.mlp_b = nn.Sequential(
        #     nn.Linear(36 * 36 * 7, 64),
        #     nn.ReLU(inplace=True),
        #     # Add more layers if needed
        # )
        # Define the MLPs for the inputs a and b
        self.mlp_a = nn.Sequential(nn.Linear(36 * 9, 512), nn.ReLU(), nn.Linear(512, 18), nn.ReLU())
        self.mlp_b = nn.Sequential(nn.Linear(36 * 36 * 7, 512), nn.ReLU(), nn.Linear(512, 18), nn.ReLU())

    def forward(self, x, a, b):
        a = a.view(a.size(0), -1)  # Flatten a
        b = b.view(b.size(0), -1)  # Flatten b
        a = self.mlp_a(a)
        b = self.mlp_b(b)

        # Reshape a and b to have an extra dimension for concatenation
        #a = a.view(a.size(0), -1, 1, 1, 1, 1)
        #b = b.view(b.size(0), -1, 1, 1, 1, 1)
        a = a.view(a.size(0), -1, 1, 1, 1)
        b = b.view(b.size(0), -1, 1, 1, 1)

        # Permute x so that the channel dimension is second
        x = x.permute(0, 4, 1, 2, 3)

        skip_connections = []

        for layer in self.encoder:
            x = layer(x)
            # Expand a and b to match the spatial dimensions of x
            a_exp = a.view(a.size(0), -1, 1, 1, 1).expand(-1, -1, *x.shape[2:])
            b_exp = b.view(b.size(0), -1, 1, 1, 1).expand(-1, -1, *x.shape[2:])

            x = torch.cat((x, a_exp, b_exp), dim=1)  # Add a and b to the U-Net input
            x = self.pointwise_conv(x)  # Reduce the number of channels with a pointwise convolution

            skip_connections.append(x)
            #x = x[:, :, ::2, ::2, ::2, :]  # Downsample in the first 3 dimensions
            x = x[:, :, ::2, ::2, ::2]  # Downsample in the first 3 dimensions

        for i, layer in enumerate(self.decoder):
            skip_x = skip_connections[-(i+1)]
            x = layer(x)
            x = torch.cat((x, skip_x), dim=1)  # Concatenate along the channel dimension

        return self.final_conv(x)
