import argparse
import os
import sys

import h5py
import keras.backend as K
import numpy as np
from Bio import SeqIO

from utils import enc_seq_onehot, enc_pssm, is_fasta, get_pssm_sequence, DeepCoil_Model, decode, SegmentResultFilter, \
    ScoreResultFilter

# cx_freeze specific
if getattr(sys, 'frozen', False):
    my_loc = os.path.dirname(os.path.abspath(sys.executable))
else:
    my_loc = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='DeepCoil')
parser.add_argument('-i',
                    help='Input file with sequence in fasta format.',
                    required=True,
                    metavar='FILE')
parser.add_argument('-out_path',
                    help='Output directory',
                    default='.',
                    metavar='DIR')
parser.add_argument('-pssm',
                    help='Use PSSM mode',
                    action='store_true')
parser.add_argument('-pssm_path',
                    metavar='DIR',
                    default='.',
                    help='Directory with PSSM files.')
parser.add_argument('-out_type',
                    help='Output type. Either \'ascii\' or \'h5\'',
                    default='ascii',
                    metavar='OUT_TYPE')
parser.add_argument('-out_filename',
                    help='Output filename. Works only with \'h5\' output mode',
                    default='out.h5',
                    metavar='OUT_FILENAME')
parser.add_argument('-skip_checks',
                    action='store_true',
                    help='Skips input verification saving some time. Use only if entirely sure or in the re-runs')
parser.add_argument('-min_residue_score',
                    default=None,
                    help="minimum score to assign residue as part of coiled coil")
parser.add_argument('-min_segment_length',
                    default=None,
                    help="minimum number of consecutive residues to ")
args = parser.parse_args()

# Verify whether weights files are present

for i in range(1, 6):
    if not os.path.isfile('%s/weights/final_seq_%s.h5' % (my_loc, i)) and not os.path.isfile(
            '%s/weights/final_seq_pssm_%s.h5' % (my_loc, i)):
        print("Weight files for the DeepCoil model are not available.")
        exit()

# INPUT VERIFICATION #

print()
if not args.skip_checks:
    print("Verifying input...")
    print()
else:
    print("Skipping input verification...")
    print()
# Check if input file exists
if not os.path.isfile(args.i):
    print('ERROR: Input file does not exist!')
    exit()
# Check if input is valid fasta file
if not is_fasta(args.i):
    print("ERROR: Malformed fasta file. Please check input!")
    exit()
if not os.path.isdir(args.out_path):
    print("ERROR: Output directory does not exist!")
    exit()
# Parse fasta file
input_data = list(SeqIO.parse(args.i, "fasta"))
sequences = [str(data.seq) for data in input_data]
entries = [''.join(e for e in str(data.id) if (e.isalnum() or e == '_')) for data in input_data]
if not len(entries) == len(set(entries)):
    print("ERROR: Sequence identifiers in the fasta file are not unique!")
    exit()
# Check sequence length and presence of non standard residues
aa1 = "ACDEFGHIKLMNPQRSTVWY"
if not args.skip_checks:
    for entry, seq in zip(entries, sequences):
        if not (len(seq) >= 25 and len(seq) <= 500):
            print(
                'ERROR: Not accepted sequence length (ID %s -%s). Only sequences between 30 and 500 residues are accepted!' % (
                    entry, len(seq)))
            exit()
        for aa in seq:
            if aa not in aa1:
                print("ERROR: Sequence (ID %s) contains non-standard residue (%s)." % (entry, aa))
                exit()

# PSSM SPECIFIC INPUT VERIFICATION #
if args.pssm:
    pssm_files = []
    # Check if directory exists
    if not os.path.isdir(args.out_path):
        print("ERROR: Directory with PSSM files does not exist!")
        exit()
    for entry, seq in zip(entries, sequences):
        pssm_fn = '%s/%s.pssm' % (args.pssm_path, entry)
        if not args.skip_checks:
            if not os.path.isfile(pssm_fn):
                print("ERROR: PSSM file for entry %s does not exist!" % entry)
                exit()
            if not get_pssm_sequence(pssm_fn) == seq:
                print("ERROR: Sequence in PSSM file does not match fasta sequence for entry %s!" % entry)
                exit()
            try:
                parsed_pssm = np.genfromtxt(pssm_fn, skip_header=3, skip_footer=5, usecols=(i for i in range(2, 22)))
                if not parsed_pssm.shape[0] == len(seq):
                    parsed_pssm = np.genfromtxt(pssm_fn, skip_header=3, skip_footer=3,
                                                usecols=(i for i in range(2, 22)))
                    if not parsed_pssm.shape[0] == len(seq):
                        raise ValueError
            except ValueError:
                print("ERROR: Malformed PSSM file for entry %s!" % entry)
                exit()
        pssm_files.append(pssm_fn)

# Encode sequence into vector format
enc_sequences = []
c = 0
len_all = len(entries)
if args.pssm:
    for seq, pssm_fn in zip(sequences, pssm_files):
        c += 1
        sys.stdout.write("\rEncoding sequences: {0} / {1}...".format(c, len_all))
        sys.stdout.flush()
        enc_sequences.append(np.concatenate((enc_seq_onehot(seq, pad_length=500),
                                             enc_pssm(pssm_fn, len(seq), pad_length=500)), axis=1))
    model = DeepCoil_Model(40)
else:
    for seq in sequences:
        c += 1
        sys.stdout.write("\rEncoding sequences: {0} / {1}...".format(c, len_all))
        sys.stdout.flush()
        enc_sequences.append(enc_seq_onehot(seq, pad_length=500))
    model = DeepCoil_Model(20)

enc_sequences = np.asarray(enc_sequences)
print()
print()
print("Predicting...")
print()

ensemble_results = {}

for i in range(1, 6):
    if args.pssm:
        model.load_weights('%s/weights/final_seq_pssm_%s.h5' % (my_loc, i))
    else:
        model.load_weights('%s/weights/final_seq_%s.h5' % (my_loc, i))
    print("Prediction part %s/5..." % i)
    predictions = model.predict(enc_sequences, verbose=1)
    print()
    decoded_predictions = [decode(pred, encoded_seq) for pred, encoded_seq in
                           zip(predictions, enc_sequences)]
    for decoded_prediction, entry in zip(decoded_predictions, entries):
        if i == 1:
            ensemble_results[entry] = decoded_prediction
        else:
            ensemble_results[entry] = np.vstack((ensemble_results[entry], decoded_prediction))
K.clear_session()
print("Writing output...")
if args.out_type == 'ascii':
    for entry, seq in zip(entries, sequences):
        f = open('%s/%s.out' % (args.out_path, entry), 'w')
        final_results = np.average(ensemble_results[entry], axis=0)
        res_filter = None
        if args.min_residue_score:
            res_filter = ScoreResultFilter(final_results, args.min_residue_score)
            res_filter.write_results(entry, seq,
                                     os.path.join(args.out_path, 'residue_filter_{}'.format(args.min_residue_score)))
        if args.min_segment_length:
            seg_filter = SegmentResultFilter(final_results, args.min_segment_length, other_filter=res_filter)
            seg_filter.write_results(entry, seq,
                                     os.path.join(args.out_path, 'segment_filter_{}'.format(args.min_segment_length)))
        for aa, prob in zip(seq, final_results):
            f.write("%s %s\n" % (aa, "% .3f" % prob))
    f.close()
elif args.out_type == 'h5':
    f = h5py.File(args.out_filename, 'w')
    for entry, seq in zip(entries, sequences):
        final_results = np.average(ensemble_results[entry], axis=0)
        f.create_dataset(data=final_results, name=entry)
    f.close()
print()
print("Done!")
