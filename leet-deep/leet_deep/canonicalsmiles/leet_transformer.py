from torch import nn
#from torch.nn import Transformer
import math


class CustomTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        # self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Position-wise Feed-Forward Networks
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Position-wise Feed-Forward Networks
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.layers = nn.ModuleList([
            CustomTransformerLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        intermediate_outputs = []
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            intermediate_outputs.append(output)

        return output, intermediate_outputs


class CustomTransformerLayerDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        # self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Position-wise Feed-Forward Networks
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross-attention
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Position-wise Feed-Forward Networks
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class CustomTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.layers = nn.ModuleList([
            CustomTransformerLayerDecoder(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt

        intermediate_outputs = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
            intermediate_outputs.append(output)

        return output, intermediate_outputs


class LeetTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.encoder = CustomTransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation)
        self.decoder = CustomTransformerDecoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout, activation)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory, encoder_intermediate_outputs = self.encoder(src, src_mask, src_key_padding_mask)
        output, decoder_intermediate_outputs = self.decoder(tgt, memory, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)

        return output, encoder_intermediate_outputs, decoder_intermediate_outputs
