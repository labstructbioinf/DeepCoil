import subprocess
import sys
import numpy as np
import h5py

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

# Test DeepCoil_PSSM binary output
cmd = "python3.5 deepcoil.py  -i test/test_3/test.fasta -pssm -pssm_path test/test_3/ -out_type h5 -out_filename test/test_3/test.h5"
code = run(cmd)
if code != 0:
    print("DeepCoil_PSSM binary output test failed!")
    EXIT = 1
f_test = h5py.File('test/test_3/test.h5', 'r')
f_bk = h5py.File('test/test_3/test.h5.bk', 'r')
for entry in ('test1', 'test1'):
    results = f_test[entry][:]
    results_bk = f_bk[entry][:]
    if not np.allclose(results, results_bk):
        print("DeepCoil_PSSM binary output test failed!")
        EXIT = 1
f_test.close()
f_bk.close()


sys.exit(EXIT)

