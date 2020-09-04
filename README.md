# **DeepCoil** #
[![DOI:10.1093/bioinformatics/bty1062 ](https://zenodo.org/badge/DOI/10.1093/bioinformatics/bty1062.svg)](https://doi.org/10.1093/bioinformatics/bty1062 )
![build](https://github.com/labstructbioinf/DeepCoil/workflows/deepcoil/badge.svg) 

**Fast and accurate prediction of coiled coil domains in protein sequences.**
## **Installation** ##
The most convenient way to install **DeepCoil** is to use pip:
```bash
$ pip install deepcoil
```

## **Usage** ##

##### Running DeepCoil as standalone:

```bash
deepcoil [-h] -i FILE [-out_path DIR]
```
| Option        | Description |
|:-------------:|-------------|
| **`-i`** | Input file in FASTA format. Can contain multiple entries. |
| **`-out_path`** | Directory where the predictions are saved. For each entry in the input file one file will be saved.|
| **`--gpu`** | Flag for turning on the GPU usage. Results in faster inference on large datasets.|


##### Running DeepCoil within script:

```python
from deepcoil import DeepCoil
from deepcoil.utils import plot_preds
from Bio import SeqIO

dc = DeepCoil(use_gpu=True)

inp = {str(entry.id): str(entry.seq) for entry in SeqIO.parse('example/example.fas', 'fasta')}

results = dc.predict(inp)

plot_preds(results['3WPA_1'], to_file='example/example.png')
```
###### Example graphical output:
![Example](example/example.png)
