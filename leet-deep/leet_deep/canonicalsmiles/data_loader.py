import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class DataLoader:
    def __init__(self, file_name):
        self.file_name = file_name
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
        df = pd.read_csv(self.file_name) #pd.read_csv('C:\\dev7\\leet\\smilesdata_b_025.csv', header=None)
        df.columns = ['input1', 'input2', 'output']

        # Define the alphabet
        self.alphabet = '@ABCFHIKNOPST[\]ac#e()i+,l-n./o12r3s458xy='
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

        # Split into training and validation
        self.train_df, self.val_df = train_test_split(df2, test_size=0.2, random_state=42)

