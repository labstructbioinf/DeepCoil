#!/bin/bash
python3.5 deepcoil.py -i example/test.fasta  -out_path example/out_seq/
python3.5 deepcoil.py -i example/test.fasta  -pssm -pssm_path example/ -out_path example/out_pssm
