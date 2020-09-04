import numpy as np


class SeqVecEncoder:

    def __init__(self, seqvec_dir=None, pad_length=100):
        """
        Encoder for SeqVec to use with the generators in the mltoolbox.keras.generators. Uses precalculated raw numpy
        arrays stored in the 'seqvec_dir'.
        :param seqvec_dir: Directory storing encoded seqvec numpy arrays.
        """
        self.seqvec_dir = seqvec_dir
        self.pad_length = pad_length

    def _encode_seqvec(self, _id, beg, end, pad_left=0):
        fn = '{}/{}.npy'.format(self.seqvec_dir, _id)
        matrix = np.load(fn)[beg:end, :]
        pad_matrix = np.zeros((self.pad_length, 1024))
        pad_matrix[pad_left:matrix.shape[0] + pad_left, 0:matrix.shape[1]] = matrix
        return pad_matrix

    def encode_batch(self, batch_df, i):
        X = np.asarray(
            [self._encode_seqvec(value['id'], value['beg'], value['end'], pad_left=value['pad_left']) for key, value in
             batch_df.iterrows()])
        return X


class SeqVecMemEncoder:

    def __init__(self, seqvec_dict=None, pad_length=100):
        """
        Encoder for SeqVec to use with the generators in the mltoolbox.keras.generators. Uses precalculated raw numpy
        arrays stored in the 'seqvec_dir'.
        :param seqvec_dir: Directory storing encoded seqvec numpy arrays.
        """
        self.seqvec_dict = seqvec_dict
        self.pad_length = pad_length

    def _encode_seqvec(self, _id, beg, end, pad_left=0):
        matrix = self.seqvec_dict[_id][beg:end, :]
        pad_matrix = np.zeros((self.pad_length, 1024))
        pad_matrix[pad_left:matrix.shape[0] + pad_left, 0:matrix.shape[1]] = matrix
        return pad_matrix

    def encode_batch(self, batch_df, i):
        X = np.asarray(
            [self._encode_seqvec(value['id'], value['beg'], value['end'], pad_left=value['pad_left']) for key, value in
             batch_df.iterrows()])
        return X
