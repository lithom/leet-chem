import torch
import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class MoleculeTransformer(nn.Module):
    def __init__(self, num_tokens, num_atoms, num_atoms_all, max_distance, d_model, nhead, num_layers, dim_feedforward):
        super(MoleculeTransformer, self).__init__()

        self.num_atoms = num_atoms
        self.num_atoms_all = num_atoms_all
        self.dim_model = d_model
        self.max_distance = max_distance

        # First layer (embedding)
        self.embedding = nn.Embedding(num_tokens, d_model)

        # Positional encoding
        self.positional_encoder = PositionalEncoding(
            dim_model=self.dim_model, dropout_p=0.01, max_len=2048 #max_len=5000
        )

        # Define the transformer encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

        #self.reduce_dim = nn.Linear(d_model, d_model // 8)  # additional layer to reduce dimension
        #self.relu = nn.ReLU()  # activation function
        #self.expander = nn.Linear(d_model * num_tokens // 8, num_atoms * num_atoms * max_distance)  # reduced number of parameters
        #self.reduction_a = nn.Linear(num_tokens,max_distance)
        #self.expander = nn.Linear(d_model,num_atoms*num_atoms)

        # Define the output layer that maps the transformer's output to the number of classes
        #self.out = nn.Linear(d_model*num_tokens, num_atoms*num_atoms*max_distance)  # 16 classes for distances 0-15
        self.out = nn.Linear(d_model * num_tokens, num_atoms * num_atoms * max_distance)  # 16 classes for distances 0-15

        #self.out_2 = nn.Linear(d_model * num_tokens, num_atoms_all * 2 * max_distance)
        self.out_2 = nn.Linear( d_model , max_distance)
        # New layer to reshape the output
        #self.expander = nn.Linear(d_model, 32 * 32)

        # A list to store intermediate layer outputs
        # self.intermediate_outputs = []

    def forward(self, x):

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        x = self.embedding(x)  #* math.sqrt(self.dim_model)
        #tgt = self.embedding(tgt) * math.sqrt(self.dim_model)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        x = x.permute(1, 0, 2)

        x = self.positional_encoder(x)
        #tgt = self.positional_encoder(tgt)


        # Reset the intermediate outputs
        # self.intermediate_outputs = []

        # Pass the input through each layer of the transformer

        x = self.transformer(x)
        #for layer in self.transformer.layers:
        #    x = layer(x)
        #    self.intermediate_outputs.append(x)

        #x = x.permute(1, 2, 0)  # switch dimensions, num_token last
        #x = self.reduction_a(x)
        #x = self.relu(x)

        #x = x.permute(0, 2, 1)  # switch dimensions, model-dim last
        #x = self.reduce_dim(x)

        #x = x.flatten(1);
        #x = self.expander(x)
        #x = x.permute(0, 2, 1)  # switch dimensions, num_token last
        #x = x.view(-1, self.num_atoms * self.num_atoms, self.max_distance)


        # Permute back the dimensions before passing through the output layer
        x = x.permute(1, 0, 2)

        # Pass the final output through the output layer
        #x = self.out(x)
        #x = self.out(x.reshape(x.size(0), -1)).view(x.size(0), self.num_atoms*self.num_atoms, self.max_distance)  # replace 32 with actual num_atoms
        x_1 = self.out(x.reshape(x.size(0), -1)).view(x.size(0), self.num_atoms * self.num_atoms, self.max_distance)  # replace 32 with actual num_atoms

        #x_2_input = self.intermediate_outputs[3].permute(1,0,2)
        x_2_input = x

        #x_2 = self.out_2(x_2_input.reshape(x.size(0), -1)).view(x.size(0), self.num_atoms_all * 2, self.max_distance)  # replace 32 with actual num_atoms
        x_2 = self.out_2(x_2_input)

        return x_1 , x_2



class Molecule3DTransformer2(nn.Module):
    def __init__(self, vocab_smiles):
        super(Molecule3DTransformer2, self).__init__()

        self.embedding    = torch.nn.Embedding(vocab_smiles,32)
        self.embedding3d  = torch.nn.Linear(4,64)

        # Sequence 1 processing
        #self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #self.conv2 = nn.Conv2d(32, 128, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.relu4 = torch.nn.ReLU()
        self.relu5 = torch.nn.ReLU()
        self.relu6 = torch.nn.ReLU()
        self.relu7 = torch.nn.ReLU()

        self.tanh1  = torch.nn.Tanh()

        # Transformer
        self.transformer = torch.nn.Transformer(d_model=192, nhead=8,num_encoder_layers=2,num_decoder_layers=1)

        # reduce dimensionality
        #self.reduce1 = nn.Linear(3876,60)
        self.reduce1 = nn.Linear(448, 96)

        self.dropout_a = nn.Dropout(0.05)
        self.dropout_b = nn.Dropout(0.05)
        #self.dropout_c = nn.Droupout(0.2)

        self.reduce_data_out        = nn.Linear(544,256)
        #self.reduce_data_additional = nn.Linear(544,256)
        self.process_seq1_a = nn.Linear(480,256)
        self.process_seq1_b = nn.Linear(256,128)
        self.process_seq1_c = nn.Linear(256,64)
        self.process_seq3   = nn.Linear(320,64)


        #self.relu_out_a = nn.ReLU()
        #self.relu_out_b = nn.ReLU()
        #self.relu_out_c = nn.ReLU()
        #self.relu_out_d = nn.ReLU()
        self.process_out_a = nn.Linear(192,3)
        #self.process_out_b = nn.Linear(16,3)
        #self.process_out_c = nn.Linear(32+3,16)
        #self.process_out_d = nn.Linear(16+3,3)


        # Output Layer
        self.out = nn.Linear(128, 3)

    def forward(self, smiles, data_out, data_additional, coords_blinded, blinding):
        smiles_embedded = self.embedding(smiles)
        data_out_a = self.relu1(self.reduce_data_out(data_out))
        seq1 = torch.cat( (smiles_embedded,data_additional,data_out_a),2)
        seq1 = self.dropout_a( self.relu3(self.process_seq1_a(seq1)) )
        seq1 = self.relu2(self.process_seq1_b(seq1))
        #seq1 = self.relu2(seq1)
        seq1 = seq1.view(seq1.size()[0], -1, 32)
        seq1 = seq1.permute(0, 2, 1)
        seq2 = torch.cat( (blinding.unsqueeze(2), coords_blinded.permute(0,2,1)),dim=2)
        seq2 = self.relu6(self.embedding3d(seq2))
        seq3 = torch.cat( (seq1 , seq2) , dim=2 )
        seq3 = self.relu5(self.process_seq3(seq3))
        seq1 = self.relu4(self.process_seq1_c(seq1))


        #self.forward(seq1,seq2)
        # seq1: 64x224 , seq2: 32x4
        #seq1 = torch.unsqueeze(seq1,1)

        #x_reduced = self.relu1( self.reduce1(seq1) )
        #x_reduced = seq1.permute(0,2,1)

        # Sequence 1 processing
        #x = self.conv1( seq1 )
        #x = nn.functional.relu(x)
        #x = self.conv2(x)
        #x = nn.functional.relu(x)

        # Adjust dimensions
        #x = x.view(x.size(0), 32, -1)
        #x_reduced = self.reduce1(x)

        # Concatenation
        x = torch.cat((seq1, seq2, seq3), dim=2)

        # Transformer
        x = x.permute(1,0,2)
        x = self.transformer(x,x)
        x = x.permute(1,0,2)

        # Output
        x = self.tanh1( self.process_out_a ( x ) )
        #coords_blinded_permuted = coords_blinded.permute(0,2,1)
        #x = self.dropout_b( self.relu_out_a( self.process_out_a ( x ) ) )
        #x = self.relu_out_a(self.process_out_a(x))
        #x = self.relu_out_b(self.process_out_b(x ))

        #x = self.dropout_b( self.relu_out_a( self.process_out_a ( torch.cat( (x , coords_blinded_permuted),dim=2))) )
        #x = self.relu_out_b(self.process_out_b(torch.cat((x, coords_blinded_permuted),dim=2)))
        #x = self.relu_out_c(self.process_out_c(torch.cat((x, coords_blinded_permuted),dim=2)))
        #x = self.relu_out_d(self.process_out_d(torch.cat((x, coords_blinded_permuted),dim=2)))

        return x.permute(0,2,1)

class Molecule3DTransformer(nn.Module):
    def __init__(self, smiles_vocab_size, d_model, max_seq_len, additional_data_size,
                 n_output_points=3 * 32):
        super().__init__()
        self.d_model = d_model
        #self.encoder = nn.TransformerEncoder(
        #    TransformerEncoderLayer(d_model, nhead, dim_feedforward), num_encoder_layers)
        self.init_pos_encodings(5000) # because we use the same value in the model A pos encoding..
        self.embedding = nn.Embedding(smiles_vocab_size, d_model)
        #self.seq_length =
        #self.transformer = nn.Transformer(196+d_model,8,3,3,dim_feedforward=2048,dropout=0.1)
        self.transformer = nn.Transformer(d_model, 8, 3, 3, dim_feedforward=2048, dropout=0.1)
        self.ingress = nn.Linear(196+d_model+3+1,d_model) # full dimension of everything concatenated
        self.dense1A = nn.Linear(max_seq_len * d_model + additional_data_size + 3 * 32 + 32, 4096)
        self.dense2 = nn.Linear(4096, n_output_points)
        self.dense3 = nn.Linear(4096, n_output_points)
        self.relu1  = nn.ReLU()
        self.relu2  = nn.ReLU()
        self.tanh = nn.Tanh()

    def init_pos_encodings(self, max_seq_len):
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * -(torch.log(torch.tensor(10000.0)) / self.d_model))
        pos_encodings = torch.empty(max_seq_len, self.d_model)
        pos_encodings[:, 0::2] = torch.sin(position * div_term)
        pos_encodings[:, 1::2] = torch.cos(position * div_term)
        self.pos_encodings = pos_encodings
        self.pos_encodings = nn.Parameter(pos_encodings, requires_grad=False)

    def forward(self, smiles_data, additional_data, coords, blinding):
        smiles_data = self.embedding(smiles_data)
        # Add position encodings to the SMILES data and flatten it
        x = (smiles_data + self.pos_encodings[0:64,:])
        x_transformer_in = torch.cat( ( x , additional_data , coords , blinding ), dim=1 )

        # Flatten the coords and blinding data
        coords_flat = coords.view(smiles_data.shape[0], -1)
        blinding_flat = blinding.view(smiles_data.shape[0], -1)

        x_transformer_in = torch.cat((x, additional_data),dim=1).view(256,-1,32)
        x_transformer_in = x_transformer_in.permute(1,0,2)
        x_transformer_out = self.transformer(x_transformer_in, x_transformer_in)
        x_transformer_out = x_transformer_out.permute(1,0,2)

        # Concatenate the flattened SMILES data, the additional data, and the flattened coords and blinding data
        x = torch.cat((x, additional_data, coords_flat, blinding_flat), dim=1)

        # Pass through the dense layers
        x1 = self.dense1A(x)
        x = self.relu1(x1)
        x = self.dense2(x)

        # Pass through the tanh activation function to get the final output
        # We already multiply by 0.5 to map [-1,1] to [-0.5,0.5] which is the same
        # normalization as the input data
        output = 0.5 * self.tanh(x).view(smiles_data.shape[0], 3, -1)

        return output

