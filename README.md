# Protein2SMILES Transformer
Protein2SMILES Transformer is De novo drug discovery of protein-specific using Transformer Neural Network, as described in the "[Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)" paper, to generate novel drugs specific to proteins.

<p align="center"><img src="https://user-images.githubusercontent.com/86023602/226380074-5b5ba664-8746-4ce7-84ca-31bfee4ca1f0.png" width="400px" /></p>

## Introduction

Protein2SMILES Transformer is a de novo drug discovery approach that generates SMILES strings, a text-based representation of a molecule, for specific protein targets. The model is trained on a large database of molecules and protein sequences collected from [Bindingdb](https://www.bindingdb.org/), and can generate new molecules that are optimized for binding to a target protein.

## Dataset Availability

All datasets used in this project are available in my Google Drive. You can access them using the following links:

- [Protein Dataset](https://drive.google.com/file/d/1crds9AKCmpX3yW1Y8z2vvL3gE6Z_ecez/view?usp=sharing)
- [SMILES Dataset](https://drive.google.com/file/d/1urykQtrMGUYfF_ZTJ7j1iiIFr2a406Er/view?usp=sharing)

## Requirements
Protein2SMILES Transformer requires the following dependencies:

- Python 3.7 or later
- PyTorch 1.13.1
- Torchtext 0.14.1
- NumPy 1.22.4
- PyQt5 5.15.9
- rdkit 2022.9.5

## Usage
To use Protein2SMILES Transformer, follow these steps:

1. Clone this repository to your local machine using git clone:
```sh
  $ git clone https://github.com/atilmohamine/protein2smiles-transformer.git
```

2. Install the required dependencies by running the following command:
```sh
  $ pip install -r requirements.txt
```
3. Run the predict.py script with the desired protein sequence as input. For example: 
```sh
  $ python predict.py --input MGLSDGEWQLVLNVWGKVEGARQPL
```
This will generate a SMILES string that is optimized for binding to the specified protein.

There are several key args for prediction as follows:
| Argument | Description | Default | Type |
| :-----| :---- | :---- | :---- |
| --input | Input Protein | none (required) | string |
| --vis | Molecule Visualization | True | boolean
| --max | Max generated sentence lenght | 150 | integer |
| --pad | Padding token | 1 | integer |
| --sos | SOS token | 2 | integer |
| --eos | EOS token | 3 | integer |

4. The output SMILES string can be used for further analysis, such as molecular docking or structure-based drug design.

## Citation

If you find this project useful in your research, please consider citing our paper:

[Amine, A.M.E., Fadila, A. Transformer neural network for protein-specific drug discovery and validation using QSAR. J Proteins Proteom (2023). https://doi.org/10.1007/s42485-023-00124-6](https://link.springer.com/article/10.1007/s42485-023-00124-6)

### BibTeX:
```bibtex
@article{AmineFadila2023,
  author    = {Atil Mohamed El Amine, Atil Fadila},
  title     = {Transformer neural network for protein-specific drug discovery and validation using QSAR},
  journal   = {Journal of Proteins and Proteomics},
  year      = {2023},
  doi       = {10.1007/s42485-023-00124-6}
}
```

## Licence
Protein2SMILES Transformer is released under the [MIT License](https://github.com/AtilMohAmine/protein2smiles-Transformer/blob/main/Licence).
