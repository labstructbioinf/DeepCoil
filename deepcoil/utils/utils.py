import numpy as np
from itertools import groupby
from Bio import SeqIO
from scipy.signal import find_peaks


# Adapted from https://stackoverflow.com/questions/44293407/how-can-i-check-whether-a-given-file-is-fasta
def is_fasta(filename):
    """
    Checks whether given file is in fasta format
    :param filename: name of the file to check
    :return: True/False denoting whether file is in fasta fortmat
    """
    with open(filename, 'r') as handle:
        fasta = SeqIO.parse(handle, 'fasta')
        return any(fasta)


def corr_seq(seq):
    """
    Corrects sequence by mapping non-std residues to 'X'
    :param seq: input sequence
    :return: corrected sequence with non-std residues changed to 'X'
    """
    letters = set(list('ACDEFGHIKLMNPQRSTVWYX'))
    seq = ''.join([aa if aa in letters else 'X' for aa in seq])
    return seq


def sharpen_preds(probs):
    """
    Sharpens raw probabilities returned by DeepCoil to more human-readable format
    :param probs: raw probabilities
    :return: sharpened probabilities
    """
    sharp_probs = np.zeros(len(probs))
    peaks = find_peaks(probs.flatten(), width=7, rel_height=0.6)
    for i in range(0, len(peaks[0])):

        beg = int(peaks[1]['left_ips'][i])
        end = int(peaks[1]['right_ips'][i])
        prob = max(probs[beg:end])

        for j in range(beg, end + 1):
            if prob >= 0.1:
                sharp_probs[j] = prob

    sharp_probs = sharp_probs.flatten()
    above_threshold = sharp_probs > 0
    for k, g in groupby(enumerate(above_threshold), key=lambda x: x[1]):
        if k:
            g = list(g)
            beg, end = g[0][0], g[-1][0]
            sharp_probs[beg:end] = max(sharp_probs[beg:end])

    return sharp_probs
