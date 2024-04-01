import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet4D_2(nn.Module):
    def __init__(self, num_classes):
        super(UNet4D_2, self).__init__()

        numAtoms = 10
        numBonds = 7

        # Define MLPs for the inputs a and b for each layer
        self.mlp_a1 = nn.Sequential(nn.Linear(36*numAtoms, 512), nn.ReLU(), nn.Linear(512, 4), nn.ReLU())
        self.mlp_b1 = nn.Sequential(nn.Linear(36*36*numBonds, 512), nn.ReLU(), nn.Linear(512, 4), nn.ReLU())

        self.mlp_a2 = nn.Sequential(nn.Linear(36*numAtoms, 512), nn.ReLU(), nn.Linear(512, 4), nn.ReLU())
        self.mlp_b2 = nn.Sequential(nn.Linear(36*36*numBonds, 512), nn.ReLU(), nn.Linear(512, 4), nn.ReLU())

        self.mlp_a3 = nn.Sequential(nn.Linear(36*numAtoms, 512), nn.ReLU(), nn.Linear(512, 4), nn.ReLU())
        self.mlp_b3 = nn.Sequential(nn.Linear(36*36*numBonds, 512), nn.ReLU(), nn.Linear(512, 4), nn.ReLU())

        # Define pointwise convolution layers for each encoder layer
        #self.pointwise_conv1 = nn.Conv3d(36+36, 36, 1)
        #self.pointwise_conv2 = nn.Conv3d(36+36+10, 18, 1)
        #self.pointwise_conv3 = nn.Conv3d(36+36+18+9, 13, 1)

        # Define the encoder and decoder layers
        self.encoder1 = ConvBlock(36+8, 64)
        self.encoder2 = ConvBlock(64+8, 128)
        self.encoder3 = ConvBlock(128+8, 256)

        self.decoder1 = ConvBlock(256+256, 128)
        self.decoder2 = ConvBlock(128+128, 64)
        self.decoder3 = ConvBlock(64+64, 36)

        #self.final_conv = nn.Conv3d(32, num_classes, 1)

    def forward(self, x, a, b):
        # Permute x so that the channel dimension is second
        x = x.permute(0, 4, 1, 2, 3)

        skip_connections = []

        # Encoder layer 1
        a1 = self.mlp_a1(a.view(a.size(0), -1))
        b1 = self.mlp_b1(b.view(b.size(0), -1))
        a1_a = a1.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1,-1, *x.shape[2:])
        b1_a = b1.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1,-1,*x.shape[2:])

        # x = torch.cat((x, a1.view(a1.size(0), -1, 1, 1, 1).expand(-1, -1, *x.shape[2:]), b1.view(b1.size(0), -1, 1, 1, 1).expand(-1, -1, *x.shape[2:])), dim=1)
        x = torch.cat((x, a1_a, b1_a), dim=1)
        #x = self.pointwise_conv1(x)
        x = self.encoder1(x)
        skip_connections.append(x)
        x = x[:, :, ::2, ::2, ::2]  # Downsample

        # Encoder layer 2
        a2 = self.mlp_a2(a.view(a.size(0), -1))
        b2 = self.mlp_b2(b.view(b.size(0), -1))
        a2_a = a2.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, *x.shape[2:])
        b2_a = b2.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, *x.shape[2:])
        #x = torch.cat((x, a2.view(a2.size(0), -1, 1, 1, 1).expand(-1, -1, *x.shape[2:]), b2.view(b2.size(0), -1, 1, 1, 1).expand(-1, -1, *x.shape[2:])), dim=1)
        x = torch.cat((x, a2_a, b2_a), dim=1)
        #x = self.pointwise_conv2(x)
        x = self.encoder2(x)
        skip_connections.append(x)
        x = x[:, :, ::2, ::2, ::2]  # Downsample

        if(torch.isnan(x).any()):
            print("NaN!!")

        # Encoder layer 3
        a3 = self.mlp_a3(a.view(a.size(0), -1))
        b3 = self.mlp_b3(b.view(b.size(0), -1))
        a3_a = a3.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, *x.shape[2:])
        b3_a = b3.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, *x.shape[2:])
        #x = torch.cat((x, a3.view(a3.size(0), -1, 1, 1, 1).expand(-1, -1, *x.shape[2:]), b3.view(b3.size(0), -1, 1, 1, 1).expand(-1, -1, *x.shape[2:])), dim=1)
        x = torch.cat((x, a3_a, b3_a), dim=1)
        #x = self.pointwise_conv3(x)
        x = self.encoder3(x)
        skip_connections.append(x)

        if(torch.isnan(x).any()):
            print("NaN!!")

        # Decoder layer 1
        #x = torch.nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)  # Upsample
        x = torch.cat((x, skip_connections[-1]), dim=1)
        x = self.decoder1(x)

        # Decoder layer 2
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)  # Upsample
        x = torch.cat((x, skip_connections[-2]), dim=1)
        x = self.decoder2(x)

        if(torch.isnan(x).any()):
            print("NaN!!")

        # Decoder layer 3
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)  # Upsample
        x = torch.cat((x, skip_connections[-3]), dim=1)
        x = self.decoder3(x)

        if(torch.isnan(x).any()):
            print("NaN!!")


        return x
