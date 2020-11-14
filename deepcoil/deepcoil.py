import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable logging messages from tensorflow

import warnings
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
from deepcoil.utils.seqvec import SeqVec
from deepcoil.utils.encoders import SeqVecMemEncoder
from deepcoil.utils.generators import SeqChunker
from deepcoil.utils import corr_seq
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import get_file
import tensorflow.keras.backend as K
from zipfile import ZipFile


class DeepCoil:

    def __init__(self, use_gpu=True, n_cpu=-1):
        """
        API for DeepCoil - a predictor of coiled-coil domains in protein sequences
        :param n_cpu: (int) - number of CPUs for inference without GPU. n_cpu=-1 will use all available resources.
        :param use_gpu: (bool) - flag for enabling or disabling GPU
        """

        self.use_gpu = use_gpu
        if n_cpu != -1:
            torch.set_num_threads(n_cpu)
        self._path = os.path.dirname(os.path.abspath(__file__))

        # Set the weights and model dir
        self._weights_prefix = f'{self._path}/weights/seq'
        self._model_loc = f'{self._path}/models/seq.json'
        self._n_weights = 5

        # Handle GPU config
        if self.use_gpu:
            available_devices = tf.config.list_physical_devices('GPU')
            if len(available_devices) == 0:
                raise SystemError('No GPU device is available!')
            elif len(available_devices) > 1:
                warnings.warn(f'Multiple GPU devices are available, will use device: {available_devices[0]}')
            # Set the memory limit on tf/keras prediction since SeqVec encoding will happen also on the same GPU
            tf.config.experimental.set_virtual_device_configuration(available_devices[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        else:
            # Disable GPU's if some will get automatically detected
            tf.config.set_visible_devices([], 'GPU')

        # Build DeepCoil model
        self._seqvec = self._setup_seqvec()
        self.model = self._setup_model()

    def _setup_seqvec(self):
        """
        Load SeqVec model to either GPU or CPU, if weights are not cached - download them from the mirror.
        :return: instance of deepcoil.utils.seqvec.SeqVec
        """
        seqvec_dir = f'{self._path}/weights/seqvec'
        seqvec_conf_fn = f'{seqvec_dir}/uniref50_v2/options.json'
        seqvec_weights_fn = f'{seqvec_dir}/uniref50_v2/weights.hdf5'
        if not (os.path.isfile(seqvec_conf_fn) and os.path.isfile(seqvec_weights_fn)):
            print('SeqVec weights are not available, downloading from the remote source (this\'ll happen only once)...')
            urls = ['https://rostlab.org/~deepppi/seqvec.zip', 'https://lbs.cent.uw.edu.pl/static/files/seqvec.zip']
            for url in urls:
                try:
                    seqvec_zip_fn = get_file(f'{self._path}/weights/seqvec.zip', url)
                    archive = ZipFile(seqvec_zip_fn)
                    archive.extract('uniref50_v2/options.json', seqvec_dir)
                    archive.extract('uniref50_v2/weights.hdf5', seqvec_dir)
                    break
                except:
                    print(f'Could not download SeqVec weights from url: {url}')
        if self.use_gpu:
            return SeqVec(model_dir=f'{seqvec_dir}/uniref50_v2', cuda_device=0, tokens_per_batch=8000)
        else:
            return SeqVec(model_dir=f'{seqvec_dir}/uniref50_v2', cuda_device=-1, tokens_per_batch=8000)

    def _setup_model(self):
        """
        Setup appropriate DeepCoil model based on the json file definition
        :return: loaded keras model
        """
        with open(self._model_loc) as f:
            conf_json = f.read()
        return model_from_json(conf_json)

    def _process_input(self, data):
        """
        Validate and process the input eventually filtering/correcting problematic sequences
        :param data: input data dict
        :return: pd.DataFrame with sequences and IDs
        """
        if not isinstance(data, dict):
            raise ValueError('Input data must be a dictionary with ids and sequences as keys and values!')
        valid_seqs = [isinstance(seq, str) for seq in data.values()]
        if not all(valid_seqs):
            raise ValueError('Input data must be a dictionary with ids and sequences as keys and values!')
        if len(data) == 0:
            raise ValueError('Empty dictionary was passed as an input!')
        non_std_seqs = [sequence != corr_seq(sequence) for sequence in data.values()]
        too_short_seqs = [len(sequence) < 20 for sequence in data.values()]
        if all(too_short_seqs):
            raise ValueError('Input sequence(s) are below 20aa length!')

        # Convert input dict to pd.DataFrame for easy handling
        data = pd.DataFrame.from_dict(data, orient='index', columns=['sequence'])

        if any(non_std_seqs):
            data['sequence'] = data['sequence'].apply(lambda x: corr_seq(x))
            warnings.warn('Non-standard residues detected in input data were corrected to X token.', UserWarning)

        if any(too_short_seqs):
            data = data[data['sequence'].str.len() >= 20]
            warnings.warn('Filtered out input sequences shorter than 20 residues.', UserWarning)

        return data

    def predict(self, data):
        """
        Evaluate sequence(s) with DeepCoil
        :param data: (dict) - dictionary containing ids (dict keys) sequences (dict values) to evaluate with DeepCoil
        :return:
        """

        data = self._process_input(data)

        # Initialize dict with preds per fold values which latter will be averaged
        cc_preds_per_fold = {key: [] for key in data.index.tolist()}
        hept_preds_per_fold = {key: [] for key in data.index.tolist()}
        # Encode SeqVec (#TODO: handle large inputs which may cause OOM errors)
        embeddings = self._seqvec.encode(data, to_file=False)

        # Setup generator that'll be evaluated
        seqvec_enc = SeqVecMemEncoder(embeddings, pad_length=500)
        gen = SeqChunker(data, batch_size=64, W_size=500, shuffle=False,
                         data_encoders=[seqvec_enc], data_cols=['sequence'])

        if self.use_gpu:
            # If GPU is used free mem used by SeqVec
            torch.cuda.empty_cache()

        # Predict with each of N predictors, depad predictions and average out for final output
        for i in range(1, self._n_weights+1):
            self.model.load_weights(f'{self._weights_prefix}_{i}.h5')

            # Raw predictions
            cc_preds, hept_preds = self.model.predict(gen, steps=len(gen), verbose=0, workers=1)

            # Remove padding and join predictions if in more than one chunk
            cc_preds_depadded = {key: np.concatenate([cc_preds[ix][ind[0]:ind[1]] for ix, ind in zip(*value)])
                                 for key, value in gen.indices.items()}
            hept_preds_depadded = {key: np.concatenate([hept_preds[ix][ind[0]:ind[1]] for ix, ind in zip(*value)])
                                   for key, value in gen.indices.items()}

            # Aggregate predictions for each fold
            for key, pred in cc_preds_depadded.items():
                cc_preds_per_fold[key].append(pred)
            for key, pred in hept_preds_depadded.items():
                hept_preds_per_fold[key].append(pred)

        K.clear_session()

        # Average predictions between 5 predictors from CV training
        cc_preds_avg = {key: np.average(value, axis=0).flatten() for key, value in cc_preds_per_fold.items()}
        hept_preds_avg = {key: np.average(value, axis=0) for key, value in hept_preds_per_fold.items()}
        # Final results
        results = {key: {'cc': probs, 'hept': hept_preds_avg[key]} for key, probs in cc_preds_avg.items()}
        return results
