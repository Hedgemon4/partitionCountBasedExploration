# Partition Count Based Exploration with Network Activations

Repository for count-based exploration on network activations with PQN.


### Installation Intructions

- Python 3.12
- Install dependencies with `python -m pip install -r requirements.txt`

### Usage

There are three versions of PQN in this repo:

1. `pqn_original.py`: An implementation of PQN which uses the exact same network, weight initializations, hyperparameters, etc. as the original PQN paper
2. `pqn.py`: Mostly the same as the original, but does not default to the same weight initialization and has slightly different hyperparameters.
3. `pqn_with_counts.py`: adds in one-to-many activations and count-based exploration bonuses to `pqn.py`

### Running the Code

Simply run `python pqn.py`. The hyperparameters can be changed with command line arguments with tyro. Use `python pqn.py --help` to see all the available options.
