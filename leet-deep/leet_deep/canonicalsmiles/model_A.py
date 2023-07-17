import torch
from torch import nn
from torch.nn import Transformer
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        self.transformer = Transformer(d_model=model_dim, nhead=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, vocab_size)

    def forward(self, input1, input2):
        # Embedding the inputs
        input1 = self.embedding(input1).permute(1, 0, 2)
        input2 = self.embedding(input2).permute(1, 0, 2)

        # Add positional encodings
        input1 = self.pos_encoder(input1)
        input2 = self.pos_encoder(input2)

        out = self.transformer(input1, input2)
        out = self.output_layer(out).permute(1, 0, 2)
        return out

