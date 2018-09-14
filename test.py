import subprocess
import sys
import numpy as np

EXIT = 0


def run(command):
    result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result.communicate()
    return result.returncode


# Test DeepCoil_SEQ
cmd = "python3.5 deepcoil.py -i test/test_1/test.fasta -out_path test/test_1/"
code = run(cmd)
if code != 0:
    print("DeepCoil_SEQ test failed!")
    EXIT = 1
results = np.loadtxt('test/test_1/test.out', usecols=1)
results_bk = np.loadtxt('test/test_1/test.out.bk', usecols=1)
if not np.array_equal(results, results_bk):
    print("DeepCoil_SEQ test failed!")
    EXIT = 1

# Test DeepCoil_PSSM
cmd = "python3.5 deepcoil.py -i test/test_2/test.fasta -out_path test/test_2/ -pssm -pssm_path test/test_2/"
code = run(cmd)
if code != 0:
    print("DeepCoil_PSSM test failed!")
    EXIT = 1
results = np.loadtxt('test/test_2/test.out', usecols=1)
results_bk = np.loadtxt('test/test_2/test.out.bk', usecols=1)
if not np.array_equal(results, results_bk):
    print("DeepCoil_PSSM test failed!")
    EXIT = 1


sys.exit(EXIT)

