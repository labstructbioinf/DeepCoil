#!/bin/bash
python3.5 ./../deepcoil.py -i test_2/test.fasta -out_path test_2/ -pssm -pssm_path test_2/
DIFF=$(diff test_2/test.out test_2/test.out.bk)
if [ "$DIFF" != "" ] 
then
    exit 1
fi
