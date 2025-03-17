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
├── train.py                # Main training script for sweeps
├── question1.py            # Implementation for question 1
├── question8.py            # Implementation for question 8 (comparison)
├── question10.py           # Implementation for question 10
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
```bash
git clone https://github.com/your-username/da6401_assignment1.git
cd da6401_assignment1
```

2. Install the required packages:
```bash
pip install numpy pandas matplotlib scikit-learn wandb keras tensorflow
```

## Usage

### Training a Model

To train a model with default parameters:

```bash
python train.py --wandb_entity your-username --wandb_project da6401_assignment1
```

For running question 1:
```bash
python question1.py --wandb_entity your-username --wandb_project da6401_assignment1
```

For running question 8 (comparison):
```bash
python question8.py --compare --wandb_entity your-username --wandb_project da6401_assignment1
```

For running question 10:
```bash
python question10.py --sweep --wandb_entity your-username --wandb_project da6401_assignment1
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

To run the hyperparameter sweep in train.py:

```python
# First, in the main function of train.py uncomment the following line as well as sweep config given below it in the code.
args.sweep = True
```

### Available Options

#### Optimization Functions
- SGD (Stochastic Gradient Descent)
- Momentum-based Gradient Descent
- Nesterov Accelerated Gradient (NAG)
- RMSprop
- Adam
- NAdam

#### Activation Functions
- Tanh
- Sigmoid
- ReLU
- Identity

#### Weight Initializers
- Random
- Xavier

## Results

The best configuration found during hyperparameter tuning:
- Activation: sigmoid
- Batch size: 64
- Epochs: 20
- Epsilon: 1.0000e-8
- Hidden size: 128
- Learning rate: 0.001
- Number of layers: 4
- Optimizer: nadam
- Weight initialization: Xavier
- Weight decay: 0

This configuration achieved 88.26% accuracy on the Fashion-MNIST test set.

## Loss Function Comparison

Our experiments show that Cross-Entropy loss generally outperforms Mean Squared Error for classification tasks, achieving higher accuracy and faster convergence as demonstrated in the comparative visualizations.

### WANDB LINK
https://wandb.ai/da24m002-indian-institute-of-technology-madras/da6401_a1/reports/DA6401-Deep-Learning-Assignment-1--VmlldzoxMTgxOTgxNA?accessToken=uugphpbjub5dporqdk7gonmie64v1velf82a3eq05rrdhcwdsjf3s2yw9v9dadse
###GIT
https://github.com/anjalim002/da6401_assignment1