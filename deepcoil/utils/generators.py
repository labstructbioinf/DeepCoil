import random
import pandas as pd
import numpy as np
from tensorflow import keras

class SeqChunker(keras.utils.Sequence):
    def __init__(self, data, W_size=64, batch_size=64, shuffle=True, neg_frac=1,
                 data_encoders=None, label_encoders=None, data_cols=None, label_cols=None):
        """
        Generates sequence chunks of fixed sized window with defined overlap between windows
        :param data: instance of the pandas DataFrame containing the dataset
        :param W_size: size of the sequence windows
        :param batch_size: size of the batches returned by the generator
        :param shuffle: True/False - shuffle entries between each epoch
        :param neg_frac: fraction of negatives used at each epoch. 'global_label' columns must be defined in the 'data'
        if neg_frac < 1
        :param data_encoders: list of data encoders
        :param label_encoders: list of label encoders
        :param data_cols: columns in the 'data' DataFrame that'll be used by the data_encoders
        (length of data_cols list must be equal to the length of data_encoders list)
        :param label_cols: columns in the 'data' DataFrame that'll be used by the label_encoders
        (length of label_cols list must be equal to the length of label_encoders list)
        """
        self.data = data.copy()
        self.batch_size = batch_size
        self.data_encoders = data_encoders
        self.data_cols = data_cols
        self.label_encoders = label_encoders
        self.label_cols = label_cols

        self.W_size = W_size
        self.shuffle = shuffle
        self.neg_frac = neg_frac

        # Validate input
        self._validate_input()
        # Select only neccesary data
        cols = []
        if self.neg_frac < 1:
            cols += ['global_label']
        if self.label_cols is not None:
            cols += self.label_cols
        cols += self.data_cols
        self.data = self.data[cols]
        self._assign_batches()

    def _validate_input(self):
        if not isinstance(self.data, pd.DataFrame): # Input data must be pd.DataFrame
            raise TypeError('Data must be a pandas DataFrame!')
        if not isinstance(self.data_encoders, list): # Encoders must be passed as the list
            raise ValueError('Data encoder(s) must be specified as list!')
        if len(self.data_encoders) != len(self.data_cols): # Number of data encoders must be equal to number of columns
            raise ValueError('Number of data encoders must be equal to number of data columns in the data df')
        if self.label_encoders is not None:
            if not isinstance(self.label_encoders, list): # Encoders must be passed as the list
                raise ValueError('Label encoder(s) must be specified as list!')
            if len(self.label_encoders) != len(self.label_cols): # Number of label encoders must be equal to number of columns
                raise ValueError('Number of label encoders must be equal to number of label columns in the data df')
        if self.neg_frac < 1 and 'global_label' not in self.data.columns: # 'global_label' must be passed for downsampling negatives
            raise ValueError('Dataframe must contain global_label column for neg class downsampling')
        if self.label_encoders is not None: # Check for the label format (single or seq2seq)
            self._single_label = [all(len(str(label)) == 1 for label in self.data[label_col].values) for label_col in
                                  self.label_cols]
        self._single_data = [all(len(str(inp)) == 1 for inp in self.data[data_col].values) for data_col in
                             self.data_cols]

    def _assign_batches(self):
        if self.neg_frac < 1:
            tmp_dict = self.data.loc[
                set(self.data[self.data['global_label'] == 0].sample(frac=self.neg_frac).index) | set(
                    self.data[self.data['global_label'] == 1].index)].to_dict(orient='index')
        else:
            tmp_dict = self.data.to_dict(orient='index')

        if self.label_encoders is not None:
            windowed_data = {indice[0]: indice[1:] for name, value in tmp_dict.items() for indice in
                             self._split_seq(name, data=[value[col] for col in self.data_cols],
                                             labels=[value[col] for col in self.label_cols])}
        else:
            windowed_data = {indice[0]: indice[1:] for name, value in tmp_dict.items() for indice in
                             self._split_seq(name, data=[value[col] for col in self.data_cols])}

        self.windowed_data = pd.DataFrame.from_dict(windowed_data, orient='index',
                                                    columns=['id', 'data', 'label', 'beg', 'end', 'pad_left'])

        miss_idxs = set(tmp_dict.keys()) - set(self.windowed_data.id.values)
        for idx in miss_idxs:
            tmp_dict.pop(idx)

        if self.shuffle:
            self.windowed_data = self.windowed_data.sample(frac=1)
        indices = self.windowed_data.groupby(by='id').indices
        pads_left = self.windowed_data.groupby(by='id').agg({'pad_left': list})['pad_left'].to_dict()
        self.windowed_data['len'] = self.windowed_data['end'] - self.windowed_data['beg']
        lens = self.windowed_data.groupby(by='id').agg({'len': list})['len'].to_dict()
        self.indices = {key: (
            list(indices[key]), [(pad_left, pad_left + len_) for pad_left, len_ in zip(pads_left[key], lens[key])]) for key in tmp_dict.keys()}
        self.windowed_data['batch'] = np.arange(len(self.windowed_data)) // self.batch_size

    def _split_seq(self, name, data=None, labels=None):
        seq = data[0]
        n_windows = 1 + len(seq) // self.W_size
        pad_left = (n_windows * self.W_size - len(seq)) // 2
        pad_right = (n_windows * self.W_size - len(seq)) // 2
        if (n_windows * self.W_size - len(seq)) % 2 != 0:
            pad_right += 1
        windows = list(range(-pad_left, len(seq), self.W_size))
        splitted_seq = []
        for c, i in enumerate(windows):
            beg = max(0, i)
            end = min(len(seq), i + self.W_size)
            label_ = labels
            if labels:
                label_ = [label if self._single_label[i] else label[beg:end] for i, label in enumerate(labels)]
            data_ = [dat if self._single_data[i] else dat[beg:end] for i, dat in enumerate(data)]
            if c == 0:
                splitted_seq.append(('{}_{}'.format(name, c), name, data_, label_, beg, end, pad_left))
            else:
                splitted_seq.append(('{}_{}'.format(name, c), name, data_, label_, beg, end, 0))
        return splitted_seq

    def __len__(self):
        return int(np.ceil(len(self.windowed_data) / self.batch_size))

    def __getitem__(self, idx):
        batch_data = self.windowed_data[self.windowed_data['batch'] == idx]
        X = [encoder.encode_batch(batch_data, i) for i, encoder in enumerate(self.data_encoders)]
        if self.label_encoders is not None:
            y = [encoder.encode_batch(batch_data, i) for i, encoder in enumerate(self.label_encoders)]
            return X, y
        return X
