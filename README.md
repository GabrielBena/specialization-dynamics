# Dynamics of Specialization in Neural Modules under Ressource Constraints

## Introduction

A self-contained implementation of the code needed to create and train modular neural-networks, with varying levels of structural modularity, to reproduce results and findings from https://arxiv.org/abs/2106.02626.

## Installation
You will need to create a new environement and install the dynspec package locally, which will take care of most requirements

```
conda create -n dynspec python=3.10
conda activate dynspec
pip install -e .
```

You will also need to install the correct pytorch version for your system separately, head over to https://pytorch.org/get-started/locally/

You're all set !

## Example
You will find a main example in the notebook modular_networks.ipynb. This notebook shows how to create and train architecture with varying parameters, and their resulting specialization levels and dynamics. It goes over all the main findings of the paper, in a straightforward fashion. 

