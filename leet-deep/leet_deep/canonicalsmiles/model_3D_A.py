from torch import nn
#from torch.nn import Transformer
import torch.nn.functional as F
import math
import torch


class CrossAttentionModule(nn.Module):
    def __init__(self, seq_dim, point_dim=1024, hidden_dim=256):
        super().__init__()
        # Assuming point_dim is the dimension after global max pooling in PointNet
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.point_proj = nn.Linear(point_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.output_proj = nn.Linear(hidden_dim, point_dim)

    def forward(self, points, seq_data):
        # seq_data is expected to be of shape (batch_size, seq_dim)
        # points is expected to be of shape (batch_size, point_dim)
        seq_proj = self.seq_proj(seq_data) #.unsqueeze(1)  # Add a sequence length dimension
        points_proj = self.point_proj(points) #.unsqueeze(1)
        # MultiheadAttention expects input of shape (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attention(seq_proj.permute(1, 0, 2), points_proj.permute(1, 0, 2), points_proj.permute(1, 0, 2))
        # Convert back to (batch_size, point_dim) for processing in fully connected layers
        attn_output = self.output_proj(attn_output).permute(1,0,2)
        return attn_output


class NormalPointNet(nn.Module):
    def __init__(self , dim_out):
        super(NormalPointNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, dim_out)  # Assuming 10 output classes

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # Input x: (batch_size, num_points, 3)

        # Permute to (batch_size, 3, num_points) for Conv1d
        x = x.permute(0, 2, 1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Global max pooling
        #x = torch.max(x, 2)[0]
        x = x.permute(0, 2, 1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)

        return x


class EnhancedPointNet(nn.Module):
    def __init__(self,dim_out=3,seq_dim=1024,seq_dim_internal=64,max_dim_internal=256,num_decoder_layers_seqTransformerA=4):
        super(EnhancedPointNet, self).__init__()

        self.ingestion_pointnet = NormalPointNet(seq_dim_internal)

        self.fcSeq0 = nn.Linear(seq_dim, seq_dim_internal)
        self.seqTransformerA = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=seq_dim_internal,nhead=8,dim_feedforward=4*seq_dim_internal),num_layers=num_decoder_layers_seqTransformerA)

        self.fcSeq1 = nn.Linear(seq_dim, seq_dim_internal)
        self.fcSeq2 = nn.Linear(seq_dim, seq_dim_internal)
        self.fcSeq3 = nn.Linear(seq_dim, seq_dim_internal)

        # self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        # self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        # self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.proj1 = nn.Linear(3,64)
        self.proj2 = nn.Linear(64, 128)
        self.proj3 = nn.Linear(128, max_dim_internal)

        # First cross-attention module after conv1
        self.cross_attention1 = CrossAttentionModule(seq_dim=seq_dim_internal, point_dim=64, hidden_dim=64)

        # Second cross-attention module after conv2
        self.cross_attention2 = CrossAttentionModule(seq_dim=seq_dim_internal, point_dim=128, hidden_dim=128)

        # Third cross-attention module after global max pooling
        self.cross_attention3 = CrossAttentionModule(seq_dim=seq_dim_internal, point_dim=max_dim_internal, hidden_dim=256)

        self.fc1 = nn.Linear(max_dim_internal, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, dim_out)  # Adjust this as per your task's requirement

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, seq_data, x):
        # Input x: (batch_size, num_points, 3)

        data3d_ingested = self.relu(self.ingestion_pointnet(x))

        seq_data = self.relu(self.fcSeq0(seq_data))
        seq_data = torch.cat((seq_data,data3d_ingested),1)
        seq_data_tf = seq_data.permute(1,0,2)
        seq_data_processed_tf = self.seqTransformerA(seq_data_tf,seq_data_tf)
        seq_data_processed = seq_data_processed_tf.permute(1,0,2)[:,:32,:]

        #x = x.permute(0, 2, 1)
        #x = self.relu(self.conv1(x))
        x = self.relu(self.proj1(x))

        # Apply first cross-attention here
        #x = x.permute(0, 2, 1)  # Prepare for cross-attention (batch_size, num_points, features)
        #x = x.permute(0,2,1)
        x = self.cross_attention1(x, seq_data_processed)
        #x = x.permute(0, 2, 1)  # Revert for next Conv layer
        #x = self.relu(self.conv2(x))
        x = self.relu(self.proj2(x))
        # Apply second cross-attention here
        #x = x.permute(0, 2, 1)  # Prepare for cross-attention
        x = self.cross_attention2(x, seq_data_processed)

        x = self.relu(self.proj3(x))
        #x = x.permute(0, 2, 1)  # Revert for next Conv layer
        #x = self.relu(self.conv3(x))
        #x = x.permute(0,2,1)
        #x = torch.max(x,2)[0]
        # Apply third cross-attention here
        x = self.cross_attention3(x, seq_data_processed)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)

        return x