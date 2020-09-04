import pandas as pd
import numpy as np
import os
from allennlp.commands.elmo import ElmoEmbedder


class SeqVec:

    def __init__(self, model_dir, cuda_device=-1, tokens_per_batch=16000):
        """
        Wrapper for efficient embedding of protein sequences with SeqVec (Heinzinger et al., 2019)
        :param model_dir: Directory storing SeqVec files (weights.hdf5 and options.json)
        :param cuda_device: Index of the CUDA device to use when encoding (-1 if CPU)
        :param tokens_per_batch: Number of tokens (amino acids per encoded sequence batch) - depends on available RAM
        """
        weights = model_dir + '/' + 'weights.hdf5'
        options = model_dir + '/' + 'options.json'
        self.seqvec = ElmoEmbedder(options, weights, cuda_device=cuda_device)
        self.tokens_per_batch = tokens_per_batch

    def encode(self, data, to_file=True, out_path=None, sum_axis=True, cut_out=False):
        """
        Encodes sequences stored in 'data' DataFrame
        :param data: pandas DataFrame storing sequences ('sequence' column) and optionally 'beg' and 'end' indices
        to cut out the embeddings
        :param to_file: If true save embedding for further use in 'out_path'
        :param out_path: Directory to store embeddings if to_file is True. Filenames match the indexes of the 'data'.
        :param sum_axis: Specifies whether first axis of the embedding will be summed up.
        This will results in Nx1024 embedding for a protein sequence of length N.
        :param cut_out: Optionally cut the embedding with the 'beg' and 'end' indices. Useful when calculating the
        embedding for whole sequence and cutting out only part of it. If True data must contain 'beg' and 'end' columns.
        :return results: if 'to_file' is false returns dictionary with data indexes as keys and embedding as values.
        """
        # Validate input DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Data must be a pandas DataFrame!')
        if 'sequence' not in data.columns:
            raise ValueError('DataFrame must contain sequence column!')
        if cut_out and 'beg' not in data.columns and 'end' not in data.columns:
            raise ValueError('DataFrame must contain beg and end columns to if cut_out is True!')
        if to_file and not os.path.isdir(out_path):
            raise OSError('Output directory does not exist!')

        # Process input DataFrame
        tmp_df = data.copy()
        tmp_df['seq_len'] = tmp_df['sequence'].apply(len) # Calculate length of each sequence in DataFrame
        tmp_df = tmp_df.sort_values(by='seq_len') # Sort sequences by length
        tmp_df['cum_seq_len'] = tmp_df['seq_len'].cumsum() # Calculate cumulative sequence lengths to split into batches
        tmp_df['batch'] = tmp_df['cum_seq_len'] // self.tokens_per_batch
        # Encode sequences in batches to speed up the process. Each batch contain at most 'tokens_per_batch' aa's.
        results = {}
        for batch in tmp_df['batch'].unique():
            df = tmp_df[tmp_df['batch'] == batch]
            sequences = df['sequence'].tolist()
            if cut_out:
                beg_indices = df['beg'].tolist()
                end_indices = df['end'].tolist()
            embs = self.seqvec.embed_sentences(sequences)
            # Sum first axis if specified
            if sum_axis:
                embs = [emb.sum(axis=0) for emb in embs]
            # Cut out sequence chunks if specified
            if cut_out:
                embs = [emb[beg:end] for emb, beg, end in zip(embs, beg_indices, end_indices)]
            # Save results
            for emb, _id in zip(embs,  df.index.values):
                if to_file:
                    np.save('{}/{}.npy'.format(out_path, _id), emb)
                else:
                    results[_id] = emb
        if not to_file:
            return results

