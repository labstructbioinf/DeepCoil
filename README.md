![Build Status](https://travis-ci.org/labstructbioinf/DeepCoil.svg?branch=master)
# **DeepCoil** #
Accurate prediction of coiled coil domains in protein sequences.
​
## **Installation** ##
First clone this repository:
```bash
$ git clone https://github.com/labstructbioinf/DeepCoil.git
```
Required packages to run DeepCoil are listed in the **`requirements.txt`** file.
We suggest running DeepCoil in the virtual environment:
If you don't have virtualenv installed do so:
```bash
$ pip3 install virtualenv
```
Create virtual environment and install required packages:
```bash
$ cd virtual_envs_location
$ virtualenv deepcoil_env
$ source deepcoil_env/bin/activate
$ cd DEEPCOIL_LOCATION
$ pip3 install -r requirements.txt
```
Test the installation:
```bash
$ ./run_example.sh
```
This should produce output **`example/out_pssm/GCN4_YEAST.out`** identical to **`example/out_pssm/GCN4_YEAST.out.bk`** and accordingly for the **`example/out_seq/`** directory.
​
## **Usage** ##
```bash
python3.5 deepcoil.py [-h] -i FILE [-out_path DIR] [-pssm] [-pssm_path DIR]
```
| Option        | Description |
|:-------------:|-------------|
| **`-i`** | Input file in FASTA format. Can contain multiple entries. |
| **`-pssm`** | Flag for the PSSM-mode. If enabled DeepCoil will require psiblast PSSM files in the pssm_path. Otherwise only sequence information will be used.|
| **`-pssm_path`** | Directory with psiblast PSSM files. For each entry in the input fasta file there must be a PSSM file. |
| **`-out_path`** | Directory where the predictions are saved. For each entry one file will be saved. |
| **`-out_type`** | Output type. Either **'ascii'** (default), which will write single file for each entry in input or **'h5'** which will generate single hdf5 file storing all predictions. |
| **`-out_filename`** | Works with **"-out_type h5"** option and specifies the hdf5 output filename Overrides the **-out_path** if specified. |
| **`-min_residue_score`** | Number from range <0,1>. If passed return sequences which have at least one residue with greater score |
| **`-min_segment_length`** | Number greater than 0. If passed return sequences which have segment of length at least **-min_segment_length** |

PSSM filenames should be based on the identifiers in the fasta file (only alphanumeric characters and '_'). For example if a fasta sequence is as follows:
```
>GCN4_YEAST RecName: Full=General control protein GCN4; AltName: Full=Amino acid biosynthesis regulatory protein
MSEYQPSLFALNPMGFSPLD....
```
PSSM file should be named **`GCN4_YEAST.pssm`**.
​
You can generate PSSM files with the following command (requires NR90 database):
```bash
psiblast -query GCN4_YEAST.fasta -db NR90_LOCATION -evalue 0.001 -num_iterations 3 -out_ascii_pssm GCN4_YEAST.pssm
```
In order to generate PSSM file from multiple sequence alignment (MSA) you can use this command:
```bash
psiblast -subject sequence.fasta -in_msa alignment.fasta -out_ascii_pssm output.pssm
```
