#!/bin/bash
python3.5 deepcoil.py -i example/test.fasta  -out_path example/out_seq/
python3.5 deepcoil.py -i example/test.fasta  -pssm -pssm_path example/ -out_path example/out_pssm

python3.5 deepcoil.py -i example/test.fasta  -out_type h5 -out_filename example/out_seq/GCN4_YEAST.h5
python3.5 deepcoil.py -i example/test.fasta  -pssm -pssm_path example/ -out_type h5 -out_filename example/out_pssm/GCN4_YEAST.h5
