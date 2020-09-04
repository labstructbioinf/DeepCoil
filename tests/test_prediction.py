import h5py
from sklearn.metrics import r2_score
from deepcoil import DeepCoil
from Bio import SeqIO


class TestPredictions:

    def test_seq_regression(self):
        # Load sequences and reference preds
        seq_dict = {str(entry.id): str(entry.seq) for entry in SeqIO.parse('tests/data/test_seq.fas', 'fasta')}
        f = h5py.File('tests/data/test_seq_ref.hdf5', 'r')
        
        # Predict with DeepCoil
        dc = DeepCoil(use_gpu=False)
        results = dc.predict(seq_dict)
        
        # Compare predictions with cached predictions
        assert all([r2_score(f[f'{key}_cc'][:], results[key]['cc']) > 0.95 for key in seq_dict.keys()])
        
        # Compare heptad predictions with cached predictions
        assert all([r2_score(f[f'{key}_hept'][:].flatten(), results[key]['hept'].flatten()) > 0.95 for key in
                    seq_dict.keys()])
