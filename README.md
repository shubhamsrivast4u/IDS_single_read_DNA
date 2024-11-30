# NEURON-IDS: One-Shot Neural Error Correction for DNA Storage Sequences

This repository contains the code implementation for the paper "NEURON-IDS: One-Shot Neural Error Correction for DNA Storage Sequences" by Shubham Srivastava, Krishna Gopal Benerjee, and Adrish Banerjee.  NEURON-IDS (NEUral Recurrent Optimizer Network for IDS error correction) is a novel neural framework for correcting insertion, deletion, and substitution errors in DNA storage sequences.

## Overview

NEURON-IDS uses specialized bidirectional recurrent neural networks to handle different types of DNA sequencing errors:
- Substitutions (incorrect nucleotide replacements)
- Insertions (extra nucleotides)
- Deletions (missing nucleotides)

The architecture employs parallel error-specific submodules with an innovative context fusion mechanism to effectively correct multiple simultaneous errors.

## Repository Structure

The repository contains Jupyter notebooks implementing different components of the NEURON-IDS framework:

- `substitution_d3.ipynb`: Implementation of substitution error correction for Dictionary D3 (Dictionary D1 in the paper)
- `insertion_d3.ipynb`: Implementation of insertion error correction for Dictionary D3
- `deletion_d3.ipynb`: Implementation of deletion error correction for Dictionary D3
- `IDS training_d3-originalnoise-final.ipynb`: Final combined model training for Dictionary D3
- Similar notebooks with `d4` suffix for Dictionary D4 (Dictionary D2 in the paper)


## Dictionary Information

The code uses two DNA sequence dictionaries:

### Dictionary D3 (Dictionary D1 in paper)
- 16 DNA sequences of length 10
- Minimum Hamming distance: 7
- Minimum Edit distance: 4
- Satisfies Hamming, Reverse, Reverse-complement and GC-content constraints
- Homopolymer free

### Dictionary D4 (Dictionary D2 in paper)
- 16 DNA sequences of length 14
- Minimum Hamming distance: 4
- Minimum Edit distance: 4
- Satisfies Hamming, Reverse, Reverse-complement and GC-content constraints
- Homopolymer free

## Requirements

- Python 3.7+
- PyTorch
- CUDA-enabled GPU (recommended)
- Additional requirements:
  - numpy
  - scipy
  - matplotlib
  - thop
  - dill



## Usage

1. Clone the repository
2. Install required dependencies
3. Run individual notebooks for specific error types or full model training
4. Notebooks contain detailed comments and explanations for each step



## Contact

For questions or issues, please open a GitHub issue in the repository.
