import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_preds(results, beg=0, end=-1, out_file=None, dpi=300):
    """
    Helper function for plotting DeepCoil results
    :param results: results for given entry returned by DeepCoil
    :param beg: (optional) beginning aa of the range to use for plotting (useful for long sequences to see only subset of results)
    :param end: (optional) end aa of the range to use for plotting (useful for long sequences to see only subset of results)
    :param out_file: (optional) if specified results will also be dumped to file
    :param dpi: (optional) DPI of the resulting image
    :return:
    """

    sharp_probs = sharpen_preds(results['cc'])[beg:end]
    probs, hept = results['cc'][beg:end], results['hept'][beg:end, :]

    fig, (ax2, ax1) = plt.subplots(2, gridspec_kw={'height_ratios': [1, 9]}, figsize=(9, 5))

    # Plot probs and sharpened probs
    ax1.plot(probs, linewidth=1, c='gray', linestyle='dashed')
    ax1.plot(sharp_probs, linewidth=2, c='black')

    # Set axis limits
    ax1.set_ylim(0.01, 1)
    ax1.set_xlim(0, len(probs))

    # Show grid
    ax1.grid(linestyle="dashed", color='gray', linewidth=0.5)

    # Find and set appropriate spacing of xticks given the sequence length
    spacings = [10, 25] + list(range(50, 1000, 50))
    spacing = spacings[np.argmin([abs(10 - len([i for i in range(0, len(probs), spacing)])) for spacing in spacings])]
    ticks = [i for i in range(0, len(probs), spacing)]
    labels = [i + beg for i in range(0, len(probs), spacing)]
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels)
    ax1.set_yticks([i / 10 for i in range(1, 10, 1)])
    ax1.xaxis.set_ticks_position('none')
    ax1.set_xlabel('Position in Sequence')
    ax1.set_ylabel('Coiled Coil Probablity')

    # Parse a, d heptad annotations and plot
    a, d = [], []
    for i, pr in enumerate(hept):
        if pr[1] > 0.2 or pr[2] > 0.2:
            if pr[1] > pr[2]:
                a.append(pr[1])
                d.append(0)
            else:
                d.append(-pr[2])
                a.append(0)
        else:
            a.append(0)
            d.append(0)
    kk = np.vstack((a, d))
    sns.heatmap(np.asarray(kk), cmap='bwr', vmin=-1, vmax=1, cbar=None, ax=ax2)

    # Hide bottom panel of the subplot
    for name, spine in ax2.spines.items():
        if name != 'bottom':
            spine.set_visible(True)

    # Show the ticks corresponding to the bottom panel
    ax2.xaxis.grid(linestyle="dashed", color='gray', linewidth=0.5)
    ax2.set_xticks([i for i in range(0, len(probs), 50)])
    ax2.set_yticks([0.5, 1.5])
    ax2.set_yticklabels(['\'a\' core pos.', '\'d\' core pos.'], rotation=0)
    ax2.set_xticks(ticks)
    ax2.xaxis.set_ticks_position('none')

    plt.subplots_adjust(hspace=0)
    if out_file:
        plt.savefig(out_file, dpi=dpi)
        plt.close()