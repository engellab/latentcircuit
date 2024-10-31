# LatentCircuit
## Description
Computational framework for fitting low-dimensional recurrent neural networks to heterogeneous neural activity and interpreting their parameters. Contains source code for Langdon & Engel, "Latent circuit inference from heterogeneous neural responses during cognitive tasks".

## Installation
- Clone the repository:

    git clone https://github.com/engellab/latentcircuit

- Create a new conda environment from the latentcircuitenv.yaml file using:

    conda env create -f latentcircuitenv.yaml

- Activate the environment:

    conda activate latentcircuitenv

## Getting started
The key parts of the code base are the two PyTorch modules LatentNet and Net, contained in the files latent_net.py and and net.py. The Net class implements an RNN which we use to train models on cognitive tasks. The LatentNet class implements
the latent circuit model which we fit to the responses of either a trained RNN or neural recording data.

To get started, the Tutorial notebook demonstrates how to use the LatentNet and Net classes by:
- Fitting a Net to a context-dependent decision-making task.
- Analyzing the parameters and performance of the trained Net.
- Fitting a LatentNet to the responses of the trained Net.
- Analyzing the resulting latent circuit mechanism arising from the LatentNet parameters.
