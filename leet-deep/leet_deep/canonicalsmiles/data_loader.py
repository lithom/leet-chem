import pandas as pd
import numpy as np
#import deepchem as dc
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from rdkit import Chem

class DataLoader:
    def __init__(self, file_name_data, file_name_alphabet):
        self.file_name_data = file_name_data
        self.file_name_alphabet = file_name_alphabet
        self.data = None
        self.loaded = False
        self.onehot_encoder = []
        self.train_df = []
        self.val_df = []
        self.alphabet = []
        self.char_to_int = []
        self.int_to_char = []

    def c2i(self, ci):
        return self.char_to_int[ci]

    def i2c(self, ci):
        return self.int_to_char[ci]
    def load(self):
        # Load the data
        df = pd.read_csv(self.file_name_data) #pd.read_csv('C:\\dev7\\leet\\smilesdata_b_025.csv', header=None)
        df.columns = ['input1', 'input2', 'output']

        # Define the alphabet
        with open(self.file_name_alphabet) as file:
            self.alphabet = file.read().strip()
        #self.alphabet = '@ABCFHIKNOPST[\]ac#e()i+,l-n./o12r3s458xy='
        #self.alphabet     = np.array(list(self.alphabet_pre)).reshape(-1,1)

        self.char_to_int = dict((c, i) for i, c in enumerate(self.alphabet))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.alphabet))

        # alphabet += '.xy'  # Add any other special characters

        # OneHotEncoder initialization
        rows = 32
        cols = 40
        # Create the 2D array
        array_2d = [[j for j in range(cols)] for _ in range(rows)]
        self.onehot_encoder = OneHotEncoder(sparse=False, categories=array_2d)
        #self.onehot_encoder = OneHotEncoder(sparse=False)

        df2 = DataFrame()


        int_lists = dict()
        one_hot_data = dict()


        # One-hot encode the sequences
        for column in df.columns:
            sequences = df[column].apply(list).to_list()  # Convert each sequence to a list of characters
            transformed_list = [[self.c2i(x) for x in sublist] for sublist in sequences]
            int_lists[column] = transformed_list
            #onehot_encoded = self.onehot_encoder.fit_transform(transformed_list)
            #one_hot_data[column] = list(onehot_encoded)

        # assemble dataframe:
        #df2 = pd.DataFrame( { df.columns[0]: one_hot_data[df.columns[0]] , df.columns[1]: one_hot_data[df.columns[1]] , df.columns[2]: one_hot_data[df.columns[2]] } )
        df2 = pd.DataFrame( { df.columns[0]: int_lists[df.columns[0]] , df.columns[1]: int_lists[df.columns[1]] , df.columns[2]: int_lists[df.columns[2]] } )

        #df2 = df2.iloc[0:2000,:]

        # Split into training and validation
        self.train_df, self.val_df = train_test_split(df2, test_size=0.2, random_state=42)


class DataLoaderMolNetTox21:
    def __init__(self, dataloader: DataLoader, datasets, label_idx=0):
        self.dataloader = dataloader
        self.datasets = datasets
        self.label_idx = label_idx
        self.data = None
        self.loaded = False
        self.train_df = []
        self.val_df = []
        self.alphabet = []
        self.char_to_int = []
        self.int_to_char = []

    def c2i(self, ci):
        if ci in self.char_to_int:
            return self.char_to_int[ci]
        return self.char_to_int['x']

    def i2c(self, ci):
        return self.int_to_char[ci]
    def load(self):

        #tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='Raw')
        train_dataset, valid_dataset, test_dataset = self.datasets

        # Define the alphabet

        # Convert dataset to tensors
        #inputs_train = encodeMolecules([x for x, _, _, _ in train_dataset.itersamples()])
        if False:
            alphabet_set = set()
            # Iterate over the strings in the dataset and collect unique characters
            for x, _, _, _ in train_dataset.itersamples():
                alphabet_set.update(set(x))
            self.alphabet = sorted(alphabet_set)

        self.alphabet = self.dataloader.alphabet

        labels_all = ([y[:1] for _, y, _, _ in train_dataset.itersamples()])

        self.char_to_int = dict((c, i) for i, c in enumerate(self.alphabet))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.alphabet))

        # alphabet += '.xy'  # Add any other special characters

        # OneHotEncoder initialization

        int_lists = dict()
        one_hot_data = dict()

        mask_considered = np.zeros((len(train_dataset),1))

        # encode the sequences
        cnt = 0
        transformed_mols      = np.zeros((0,40))
        transformed_mols_inp2 = np.zeros((0,40))
        for x, _, _, _ in train_dataset.itersamples():
            cnt = cnt+1
            sequence = Chem.MolToSmiles(x)  # Convert each sequence to a list of characters
            if(len(sequence)>40):
                continue
            padding_size = 40 - len(sequence)
            left_padding = 'y' * (padding_size // 2)
            right_padding = 'y' * (padding_size - (padding_size // 2))
            padded_sequence = left_padding + sequence + right_padding

            mask_considered[cnt-1]=1
            #transformed_list = [[self.c2i(x) for x in sublist] for sublist in sequence]
            transformed_list = [self.c2i(x) for x in padded_sequence]
            transformed_list_inp2 = [self.c2i(x) for x in ('x'*40)]
            transformed_mols = np.vstack((transformed_mols,np.array(transformed_list).reshape((1,40))))
            transformed_mols_inp2 = np.vstack((transformed_mols_inp2, np.array(transformed_list_inp2).reshape((1, 40))))
            #onehot_encoded = self.onehot_encoder.fit_transform(transformed_list)
            #one_hot_data[column] = list(onehot_encoded)

        # assemble dataframe:
        #df2 = pd.DataFrame( { df.columns[0]: one_hot_data[df.columns[0]] , df.columns[1]: one_hot_data[df.columns[1]] , df.columns[2]: one_hot_data[df.columns[2]] } )
        df2_mols = pd.DataFrame( { "Structures": transformed_mols[:4950].tolist() , "input2": transformed_mols_inp2[:4950].tolist() } )
        df2_labels = pd.DataFrame( data=np.concatenate(labels_all)[np.where(mask_considered.flatten()!=0)].reshape(-1,1), columns=['Label'])
        df2 = pd.concat( [df2_mols,df2_labels] , axis=1 )

        #df2 = df2.iloc[0:2000,:]

        # Split into training and validation
        self.train_df, self.val_df = train_test_split(df2, test_size=0.2, random_state=42)
        print('mkay..')
