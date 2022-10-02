import os
import argparse
from Bio import SeqIO
from deepcoil import DeepCoil
from deepcoil.utils import is_fasta, sharpen_preds, plot_preds


def main():

    parser = argparse.ArgumentParser(description='DeepCoil')
    parser.add_argument('-i',
                        help='Input file with sequence in fasta format.',
                        required=True,
                        metavar='FILE')
    parser.add_argument('-out_path',
                        help='Output directory',
                        default='.',
                        metavar='DIR')
    parser.add_argument('-n_cpu',
                        help='Number of CPUs to use in the prediction',
                        default=-1,
                        type=int,
                        metavar='NCPU')
    parser.add_argument('--gpu',
                        help='Use GPU. This option overrides -n_cpu option',
                        action='store_true')
    parser.add_argument('--plot',
                        help='Plot predictions. Images will be stored in the path defined by the -out_path',
                        action='store_true')
    parser.add_argument('--dpi',
                        help='DPI of the produced images',
                        default=300,
                        type=int,
                        metavar='DPI')
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.isfile(args.i):
        print('ERROR: Input file does not exist!')
        exit()
    # Check if input is valid fasta file
    if not is_fasta(args.i):
        print("ERROR: Malformed fasta file. Please check input!")
        exit()
    # Check if output dir exists
    if not os.path.isdir(args.out_path):
        print("ERROR: Output directory does not exist!")
        exit()

    # Verify fasta file
    raw_data = list(SeqIO.parse(args.i, "fasta"))
    data = {''.join(e for e in str(entry.id) if (e.isalnum() or e == '_')): str(entry.seq) for entry in raw_data}
    if not len(data) == len(raw_data):
        print("ERROR: Sequence identifiers in the fasta file are not unique!")
        exit()

    print("Loading DeepCoil model...")
    dc = DeepCoil(use_gpu=args.gpu, n_cpu=args.n_cpu)

    print('Predicting...')
    preds = dc.predict(data)

    print('Writing output...')

    inp_keys = set(data.keys())
    out_keys = set(preds.keys())

    if len(out_keys) < len(inp_keys):
        print('WARNING: Predictions for some sequences were not calculated due to length limitations and/or other errors.' \
    ' Inspect the warnings and results carefully!')

    for entry in out_keys:
        f = open(f'{args.out_path}/{entry}.out', 'w')
        cc_pred_raw = preds[entry]['cc']
        cc_pred = sharpen_preds(cc_pred_raw)
        hept_pred = preds[entry]['hept']
        f.write('aa\tcc\traw_cc\tprob_a\tprob_d\n')
        for aa, cc_prob, cc_prob_raw, a_prob, d_prob in zip(data[entry], cc_pred, cc_pred_raw, hept_pred[:, 1], hept_pred[:, 2]):
            f.write('{0}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}\n'.format(aa, float(cc_prob), float(cc_prob_raw), float(a_prob), float(d_prob)))
        f.close()
    if args.plot:
        for entry in out_keys:
            plot_preds(preds[entry], out_file=f'{args.out_path}/{entry}.png', dpi=args.dpi)
    print("Done!")

