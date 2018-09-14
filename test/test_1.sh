#!/bin/bash
python3.5 ./../deepcoil.py -i test_1/test.fasta -out_path test_1/
DIFF=$(diff test_1/test.out test_1/test.out.bk) 
if [ "$DIFF" != "" ] 
then
    exit 1
fi
