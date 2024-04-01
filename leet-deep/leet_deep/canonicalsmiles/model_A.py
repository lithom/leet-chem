import torch
from torch import nn
from torch.nn import Transformer
import torch.nn.functional as F
#from leet_deep.canonicalsmiles import LeetTransformer
from .leet_transformer import LeetTransformer
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

class Seq2SeqModel_2(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers):
        super(Seq2SeqModel_2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        self.transformer = LeetTransformer(d_model=model_dim, nhead=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers)#Transformer(d_model=model_dim, nhead=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, vocab_size)

    def forward(self, input1, input2):
        # Embedding the inputs
        input1 = self.embedding(input1).permute(1, 0, 2)
        input2 = self.embedding(input2).permute(1, 0, 2)

        # Add positional encodings
        input1 = self.pos_encoder(input1)
        input2 = self.pos_encoder(input2)

        out, encoder_intermediate_outputs, decoder_intermediate_outputs = self.transformer(input1, input2)
        out = self.output_layer(out).permute(1, 0, 2)

        intermediate = encoder_intermediate_outputs+decoder_intermediate_outputs
        all_intermediate_outputs = torch.cat(intermediate, dim=2).permute(1, 0, 2)
        all_intermediate_outputs = torch.flatten(all_intermediate_outputs,start_dim=1)

        return out, all_intermediate_outputs


class Seq2SeqModel_3(nn.Module):
    def __init__(self, vocab_size_in, vocab_size_out_1, dim_expansion_out_1, vocab_size_out_2, model_dim, num_layers):
        super(Seq2SeqModel_3, self).__init__()
        self.embedding = nn.Embedding(vocab_size_in, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        self.transformer = LeetTransformer(d_model=model_dim, nhead=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers)#Transformer(d_model=model_dim, nhead=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.output_layer   = nn.Linear(model_dim, vocab_size_out_1*dim_expansion_out_1)
        self.output_layer_2 = nn.Linear(model_dim, vocab_size_out_2)

    def forward(self, input1, input2):
        # Embedding the inputs
        input1 = self.embedding(input1).permute(1, 0, 2)
        input2 = self.embedding(input2).permute(1, 0, 2)

        # Add positional encodings
        input1 = self.pos_encoder(input1)
        input2 = self.pos_encoder(input2)

        out, encoder_intermediate_outputs, decoder_intermediate_outputs = self.transformer(input1, input2)

        out_1 = self.output_layer(out).permute(1, 0, 2)
        out_2 = self.output_layer_2(out).permute(1, 0, 2)

        intermediate = encoder_intermediate_outputs+decoder_intermediate_outputs
        all_intermediate_outputs = torch.cat(intermediate, dim=2).permute(1, 0, 2)
        all_intermediate_outputs = torch.flatten(all_intermediate_outputs,start_dim=1)

        return out_1, out_2, all_intermediate_outputs


# expansions_list: each element is a tuple (num_classes,dim_expansion)
class Seq2SeqModel_4(nn.Module):
    def __init__(self, vocab_size_in, expansions_list, model_dim, num_layers):
        super(Seq2SeqModel_4, self).__init__()
        self.embedding = nn.Embedding(vocab_size_in, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        self.transformer = LeetTransformer(d_model=model_dim, nhead=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers)#Transformer(d_model=model_dim, nhead=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers)

        self.output_layers = nn.ModuleList()
        for exp_vocab, exp_dim_exp in expansions_list:
            self.output_layers.append(nn.Linear(model_dim, exp_vocab*exp_dim_exp))

    def forward(self, input1, input2):
        # Embedding the inputs
        input1 = self.embedding(input1).permute(1, 0, 2)
        input2 = self.embedding(input2).permute(1, 0, 2)

        # Add positional encodings
        input1 = self.pos_encoder(input1)
        input2 = self.pos_encoder(input2)

        out, encoder_intermediate_outputs, decoder_intermediate_outputs = self.transformer(input1, input2)

        intermediate = encoder_intermediate_outputs+decoder_intermediate_outputs
        all_intermediate_outputs = torch.cat(intermediate, dim=2).permute(1, 0, 2)
        #all_intermediate_outputs = torch.flatten(all_intermediate_outputs,start_dim=1)

        out_all = []
        out_all.append(all_intermediate_outputs)

        for output_layer in self.output_layers:
            out_i = output_layer(out).permute(1, 0, 2)
            out_all.append(out_i)

        return out_all

def tuple_product(tuple):
    product = 1
    for element in tuple:
        product *= element
    return product

# expansions_list: each element is a tuple (num_classes,dim_expansion)
# outputs [1..n] are automatically shaped into class format
class Seq2SeqModel_5(nn.Module):
    def __init__(self, vocab_size_in, sequence_length, expansions_list, model_dim, num_layers):
        super(Seq2SeqModel_5, self).__init__()
        self.embedding = nn.Embedding(vocab_size_in, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim,max_len=256)
        self.transformer = LeetTransformer(d_model=model_dim, nhead=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers)#Transformer(d_model=model_dim, nhead=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers)

        self.sequence_length = sequence_length
        self.output_layers = nn.ModuleList()
        self.expansions_list = expansions_list
        for exp_vocab, exp_dimensions in expansions_list:
            out_dimension_total = tuple_product(exp_dimensions)
            if out_dimension_total % self.sequence_length != 0:
                raise ValueError("out_dimension_total % self.sequence_length != 0 ..")
            self.output_layers.append(nn.Linear(model_dim, out_dimension_total // sequence_length ))

    def forward(self, input1, input2):
        # Embedding the inputs
        input1 = self.embedding(input1).permute(1, 0, 2)
        input2 = self.embedding(input2).permute(1, 0, 2)

        # Add positional encodings
        input1 = self.pos_encoder(input1)
        input2 = self.pos_encoder(input2)

        out, encoder_intermediate_outputs, decoder_intermediate_outputs = self.transformer(input1, input2)

        intermediate = encoder_intermediate_outputs+decoder_intermediate_outputs
        all_intermediate_outputs = torch.cat(intermediate, dim=2).permute(1, 0, 2)
        #all_intermediate_outputs = torch.flatten(all_intermediate_outputs,start_dim=1)

        out_all = []
        out_all.append(all_intermediate_outputs)

        for idx, output_layer in enumerate( self.output_layers ):
            out_i = output_layer(out).permute(1, 0, 2)
            out_i = out_i.reshape( (out_i.size()[0], *(self.expansions_list[idx][1]) ) )
            out_all.append(out_i)

        return out_all

    # class PositionalEncoding(nn.Module):
    #     def __init__(self, d_model, max_len=5000):
    #         super(PositionalEncoding, self).__init__()
    #         self.d_model = d_model
    #         pe = torch.zeros(max_len, d_model)
    #         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    #         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    #         pe[:, 0::2] = torch.sin(position * div_term)
    #         pe[:, 1::2] = torch.cos(position * div_term)
    #         pe = pe.unsqueeze(0).transpose(0, 1)
    #         self.register_buffer('pe', pe)
    #
    #     def forward(self, x):
    #         x = x + self.pe[:x.size(0), :]
    #         return x

class AutoEncoderTransformerEncoder(nn.Module):
    def __init__(self, length_smiles, num_atoms, dim_smiles, d_model=32, nhead=8, num_layers=2, dropout=0.1):
        super(AutoEncoderTransformerEncoder, self).__init__()

        self.length_smiles = length_smiles
        self.num_atoms = num_atoms

        self.smiles_embedding_1 = nn.Linear(dim_smiles, d_model)
        self.smiles_embedding_2 = nn.Linear(dim_smiles, d_model)

        self.spatial_embedding_1 = nn.Linear(3, d_model)  # 3D coordinates
        self.spatial_embedding_2 = nn.Linear(2, d_model)  # "2D" coordinates
        self.pos_encoder = PositionalEncoding(d_model)

        # First Encoding Step
        encoder_layers_1 = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
        self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layers_1, num_layers)
        #self.linear_down_1_a = nn.Linear(length_smiles+num_atoms, num_atoms) # Explanation: at then end of
        self.linear_down_1_b = nn.Linear(d_model, 2)

        # Second Encoding Step
        encoder_layers_2 = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
        self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layers_2, num_layers)
        #self.linear_down_2_a = nn.Linear(length_smiles+num_atoms, num_atoms)  # Adjusted to accept output from linear_down_2
        self.linear_down_2_b = nn.Linear(d_model, 1)  # Adjusted to accept output from linear_down_2

        # LayerNorm at the end:
        self.encoder_output_norm = nn.LayerNorm(d_model)

    def forward(self, smiles_data, spatial_data):
        # two smiles embeddings for the two transformer steps..
        smiles_embedded_1 = F.relu(self.smiles_embedding_1(smiles_data))
        smiles_embedded_2 = F.relu(self.smiles_embedding_2(smiles_data))
        spatial_embedded_1 = F.relu(self.spatial_embedding_1(spatial_data))


        # convert to transformer layout
        smiles_embedded_tf_1  = smiles_embedded_1.permute(1,0,2)
        smiles_embedded_tf_2 = smiles_embedded_2.permute(1,0,2)
        spatial_embedded_tf_1 = spatial_embedded_1.permute(1,0, 2)

        combined_embedding_tf_1 = torch.cat((smiles_embedded_tf_1, spatial_embedded_tf_1), dim=0) # concatenate sequences

        embedded_pos_tf_1 = self.pos_encoder(combined_embedding_tf_1)

        # First Encoding Step
        encoded_tf_1 = self.transformer_encoder_1(embedded_pos_tf_1)
        encoded_1 = encoded_tf_1.permute(1,0,2)
        encoded_reduced = encoded_1[:,:self.num_atoms,:] # we just throw away what we dont need.. (recommended by gpt4)
        encoded_reduced_1 = self.linear_down_1_b(encoded_reduced)

        # Second Encoding Step
        spatial_embedded_2 = F.relu(self.spatial_embedding_2(encoded_reduced_1))
        spatial_embedded_tf_2 = spatial_embedded_2.permute(1,0,2)
        combined_embedding_tf_1 = torch.cat((smiles_embedded_tf_2, spatial_embedded_tf_2), dim=0)  # concatenate sequences

        encoded_2_tf = self.transformer_encoder_2(combined_embedding_tf_1)
        encoded_2 = encoded_2_tf.permute(1,0,2) # permute back into non-transformer layout

        encoded_reduced_2 = encoded_2[:,:self.num_atoms,:] # we just throw away what we dont need.. (recommended by gpt4)
        encoded_reduced_2_normalized = self.encoder_output_norm(encoded_reduced_2)
        encoded_reduced_2_final = self.linear_down_2_b(encoded_reduced_2_normalized)

        return encoded_reduced_2_final


class AutoEncoderTransformerDecoder(nn.Module):
    def __init__(self, length_smiles, num_atoms, dim_smiles, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super(AutoEncoderTransformerDecoder, self).__init__()

        self.length_smiles = length_smiles
        self.num_atoms = num_atoms
        self.smiles_embedding = nn.Linear(dim_smiles, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, d_model * 4, dropout)

        self.distance_transformer = nn.TransformerDecoder(decoder_layers, 4)

        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, 3)  # Convert output to 3D coordinates

        # To tap the distance information:
        self.distance_out = nn.Linear(d_model, 32)

    def forward(self, smiles_data, encoded_data):
        smiles_embedded = F.relu(self.smiles_embedding(smiles_data))
        # Assuming encoded_data is (batch, num_atoms, 1), we need to expand it to match the embedding size
        encoded_expanded = encoded_data.expand(-1, -1, self.smiles_embedding.out_features)

        #convert to transformer layout
        smiles_embedded_tf_1 = smiles_embedded.permute(1,0,2)
        encoded_embedded_tf_1 = encoded_expanded.permute(1,0,2)
        combined_embedding_tf_1 = torch.cat((smiles_embedded_tf_1, encoded_embedded_tf_1),dim=0)  # concatenate sequences

        combined_input_tf = self.pos_encoder(combined_embedding_tf_1)

        distance_decoded_tf = self.distance_transformer(combined_input_tf, combined_input_tf)
        combined_embedding_tf_2 = torch.cat((combined_embedding_tf_1, distance_decoded_tf),dim=0)  # concatenate sequences
        combined_input_tf_2 = self.pos_encoder(combined_embedding_tf_2)

        decoded_tf = self.transformer_decoder(combined_input_tf_2, combined_input_tf_2)

        #convert to normal layout
        decoded = decoded_tf.permute(1,0,2)
        decoded_distance = distance_decoded_tf.permute(1,0,2)

        decoded_reduced = decoded[:,:32,:]
        out = self.fc_out(decoded_reduced)
        out_distances = decoded_distance[:,:32,:32]

        return out , out_distances





class GeneralTransformerModel(nn.Module):
    def __init__(self, input_smiles_dim, input_other_dim, d_model, d_out, nhead, num_decoder_layers, dropout=0.1):
        super(GeneralTransformerModel, self).__init__()
        self.input_smiles_dim = input_smiles_dim
        self.input_other_dim = input_other_dim
        self.input_projection_smiles = nn.Linear(input_smiles_dim, d_model)
        self.input_projection_other  = nn.Linear(input_other_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout),
            num_layers=num_decoder_layers
        )
        self.fc_out = nn.Linear(d_model, d_out)  # Output size = number of elements in upper right part of the matrix

    def forward(self, x_smiles, x_other):
        x_smiles =  F.relu(self.input_projection_smiles(x_smiles))
        x_other  =  F.relu(self.input_projection_other(x_other))

        x = torch.cat( (x_smiles,x_other), dim = 1 )
        x_tf = x.permute(1,0,2)
        x_tf_pos = self.pos_encoder(x_tf)
        y_tf = self.transformer_decoder(x_tf_pos, x_tf_pos)
        y = y_tf.permute(1,0,2)
        return self.fc_out(y)
