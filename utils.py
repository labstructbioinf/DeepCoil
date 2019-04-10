import os
import sys

import keras.backend as K
import numpy as np
from Bio import SeqIO
from keras.layers import Convolution1D, Dropout, Dense
from keras.models import Sequential
from keras.regularizers import l2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
aa1 = list("ACDEFGHIKLMNPQRSTVWY")
aa_indices = {aa1[k]: k for k in range(0, len(aa1))}


# Adapted from https://stackoverflow.com/questions/44293407/how-can-i-check-whether-a-given-file-is-fasta
# Checks whether file is in fasta format
def is_fasta(filename):
    with open(filename, "r") as handle:
        fasta = SeqIO.parse(handle, "fasta")
        return any(fasta)  # False when `fasta` is empty, i.e. wasn't a FASTA file


# Reads sequence from Psiblast PSSM file
def get_pssm_sequence(fn):
    c = 0
    seq_list = []
    try:
        with open(fn) as f:
            for line in f:
                if c > 2:
                    try:
                        aa = line.split()
                        seq_list.append(aa[1])
                    except IndexError:
                        break
                c += 1
        f.close()
    except FileNotFoundError:
        pass
    return ''.join(seq_list)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Encodes amino acid sequence in one-hot format
def _enc_aa(aa):
    enc_aa = [0] * 20
    enc_aa[aa_indices[aa]] = 1
    return enc_aa


def enc_seq_onehot(seq, pad_length=None, pad_left=0):
    matrix = np.asarray([_enc_aa(aa) for aa in seq])
    if pad_length:
        pad_matrix = np.zeros((pad_length, 20))
        pad_matrix[pad_left:matrix.shape[0] + pad_left, 0:20] = matrix
        return pad_matrix
    return matrix


# Encodes PSSM
def enc_pssm(pssm_file, length, pad_length=None, pad_left=0):
    pssm_matrix = sigmoid(np.genfromtxt(pssm_file, skip_header=3, skip_footer=5, usecols=(i for i in range(2, 22))))
    if pssm_matrix.shape[0] != length:
        pssm_matrix = sigmoid(
            np.genfromtxt(pssm_file, skip_header=3, skip_footer=3, usecols=(i for i in range(2, 22))))  # Custom PSSM
    if pad_length:
        pad_matrix = np.zeros((pad_length, 20))
        pad_matrix[pad_left:pssm_matrix.shape[0] + pad_left, 0:pssm_matrix.shape[1]] = pssm_matrix
        return pad_matrix
    return pssm_matrix


# Decodes predictions (takes into the account padding of sequence)
def decode(pred, enc_seq):
    return np.delete(pred, np.where(~enc_seq.any(axis=1)), axis=0)[:, 1]


# DeepCoil model architecture
def DeepCoil_Model(inp_length):
    model = Sequential()
    model.add(
        Convolution1D(64, 28, padding='same', activation='relu', kernel_regularizer=l2(0.0001),
                      input_shape=(500, inp_length)))
    model.add(Dropout(0.5))
    model.add(Convolution1D(64, 21, padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))
    model.add(Dense(128))
    model.add(Dense(2, activation='softmax', name='out'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


class ResultFilter:
    def __init__(self, score_array, filter_value, other_filter=None):
        self.score_array = np.asarray(score_array)
        self.filter_value = filter_value
        self.other_filter = other_filter
        self.is_correct = False
        if self.other_filter:
            self.score_array = self.other_filter.apply()
        self.indices_column = np.zeros(len(self.score_array))

    def apply(self):
        if self.filter_value <= 0:
            self.is_correct = True
            return self.score_array

    def write_results(self, seq_id, seq, output_path='.', sufix='_filtered'):
        lines = []
        self.apply()
        if self.is_correct:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for i, j in zip(seq, self.indices_column):
                lines.append(" ".join([i, str(int(j))]))
            with open(os.path.join(output_path, seq_id + sufix), 'w') as n:
                n.write("\n".join(lines))


class ScoreResultFilter(ResultFilter):
    def __init__(self, score_array, filter_value, other_filter=None):
        super().__init__(score_array, filter_value, other_filter)

    def apply(self):
        try:
            self.filter_value = float(self.filter_value)
        except:
            pass
        if self.filter_value > 1 or self.filter_value < 0:
            return self.indices_column
        result = np.where(self.score_array >= self.filter_value)[0]
        self.indices_column[result] = 1
        self.is_correct = bool(len(result))
        return self.indices_column

    def write_results(self, seq_id, seq, output_path='.', sufix='_filtered'):
        super().write_results(seq_id, seq, output_path, sufix='_filtered_residues')


class SegmentResultFilter(ResultFilter):

    def __init__(self, score_array, filter_value, other_filter=None):
        if not other_filter:
            other_filter = ScoreResultFilter(score_array, 0.5)
            other_filter.apply()
        super().__init__(score_array, filter_value, other_filter)

    def apply(self):

        try:
            self.filter_value = int(self.filter_value)
        except:
            pass
        if not isinstance(self.filter_value, int):
            return self.indices_column
        i = 0
        cnt = 0
        while i < len(self.score_array) + 1:
            if i == len(self.score_array) or self.score_array[i] == 0:
                if cnt and i - cnt >= self.filter_value:
                    self.indices_column[cnt:i] = 1
                    self.is_correct = True
                cnt = 0
            else:
                if not cnt:
                    cnt = i
            i += 1
        return self.indices_column

    def write_results(self, seq_id, seq, output_path='.', sufix='_filtered_segments'):
        super().write_results(seq_id, seq, output_path, sufix='_filtered_residues')


# Exit function
def exit():
    print("Run failed!")
    K.clear_session()
    sys.exit(1)
