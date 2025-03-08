# DA6401 Assignment 1: Neural Network Implementation with Backpropagation

This repository contains a from-scratch implementation of a feedforward neural network with backpropagation for the Fashion-MNIST classification task, as required for the DA6401 Assignment 1.

## Features

- Custom implementation of feedforward neural networks with flexible architecture
- Various activation functions: sigmoid, tanh, ReLU, identity
- Multiple optimizers implemented: SGD, Momentum, NAG, RMSprop, Adam, NAdam
- Loss functions: Cross-Entropy and Mean Squared Error
- Weight initialization techniques: Random and Xavier
- Integration with Weights & Biases for experiment tracking

## Project Structure

```
da6401_assignment1/
├── README.md               # Project documentation
├── train.py                # Main training script
├── sweeps/                 # Sweep configuration files
│   └── sweep_config.yaml   # Sweep configuration
└── src/                    # Source code directory
    ├── __init__.py
    ├── data.py             # Data loading and preprocessing
    ├── activations.py      # Activation functions
    ├── layers.py           # Neural network layers
    ├── losses.py           # Loss functions
    ├── network.py          # Neural network architecture
    └── optimizers.py       # All optimizers
```

## Installation

1. Clone the repository:
```
git clone https://github.com/your-username/da6401_assignment1.git
cd da6401_assignment1
```

2. Install the required packages:
```
pip install numpy pandas matplotlib scikit-learn wandb keras tensorflow
```

## Usage

### Training a Model

To train a model with default parameters:

```
python train.py --wandb_entity your-username --wandb_project da6401_assignment1
```

### Command Line Arguments

```
python train.py --help
```

Key arguments:
- `-wp`, `--wandb_project`: Project name for Weights & Biases
- `-we`, `--wandb_entity`: Username for Weights & Biases
- `-d`, `--dataset`: Dataset to use (fashion_mnist or mnist)
- `-e`, `--epochs`: Number of training epochs
- `-b`, `--batch_size`: Batch size for training
- `-l`, `--loss`: Loss function (cross_entropy or mean_squared_error)
- `-o`, `--optimizer`: Optimizer (sgd, momentum, nag, rmsprop, adam, nadam)
- `-lr`, `--learning_rate`: Learning rate
- `-nhl`, `--num_layers`: Number of hidden layers
- `-sz`, `--hidden_size`: Size of hidden layers
- `-a`, `--activation`: Activation function (sigmoid, tanh, ReLU, identity)

### Running Hyperparameter Sweeps

To run a hyperparameter sweep:

```
# First, initialize the sweep
wandb sweep sweeps/sweep_config.yaml

# Then run the sweep agent (replace SWEEP_ID with the ID from previous command)
wandb agent your-username/da6401_assignment1/SWEEP_ID
```

## Results

The best configuration found during hyperparameter tuning:
- Number of hidden layers: 3
- Size of hidden layers: 128
- Activation function: ReLU
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Weight initialization: Xavier

This configuration achieved 92% accuracy on the Fashion-MNIST test set.

## Self Declaration

I, [Your Name], swear on my honour that I have written the code and the report by myself and have not copied it from the internet or other students.
