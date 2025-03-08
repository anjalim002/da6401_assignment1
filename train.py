print("🚀 train.py has started execution!")

import argparse
import wandb

print("✅ Imports successful")

# Existing code...

import argparse
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.data import load_data, split_train_val, get_batches
from src.network import NeuralNetwork
from src.losses import get_loss
from src.optimizers import get_optimizer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network')
    
    # Wandb arguments
    parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default='myname',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    
    # Dataset arguments
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',
                        choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use')
    
    # Training arguments
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='Number of epochs to train neural network')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size used to train neural network')
    
    # Loss arguments
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['mean_squared_error', 'cross_entropy'],
                        help='Loss function to use')
    
    # Optimizer arguments
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimizer to use')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='Momentum used by momentum and nag optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.9,
                        help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9,
                        help='Beta1 used by adam and nadam optimizers')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999,
                        help='Beta2 used by adam and nadam optimizers')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-8,
                        help='Epsilon used by optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0,
                        help='Weight decay used by optimizers')
    
    # Model arguments
    parser.add_argument('-w_i', '--weight_init', type=str, default='Xavier',
                        choices=['random', 'Xavier'],
                        help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3,
                        help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', type=int, default=128,
                        help='Number of hidden neurons in a feedforward layer')
    parser.add_argument('-a', '--activation', type=str, default='ReLU',
                        choices=['identity', 'sigmoid', 'tanh', 'ReLU'],
                        help='Activation function to use')
    
    return parser.parse_args()

def accuracy(predictions, targets):
    """
    Compute accuracy
    
    Args:
        predictions: numpy array of shape (batch_size, num_classes)
        targets: numpy array of shape (batch_size, num_classes)
        
    Returns:
        accuracy: scalar
    """
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(targets, axis=1)
    
    return np.mean(predicted_labels == true_labels)

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save confusion matrix
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return 'confusion_matrix.png'

def train_model(args):
    """
    Train neural network
    
    Args:
        args: Arguments
    """
    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    config = wandb.config
    
    # Load data
    (X_train, y_train_one_hot, y_train), (X_test, y_test_one_hot, y_test) = load_data(config.dataset)
    
    # Split training data into training and validation sets
    (X_train, y_train_one_hot, y_train), (X_val, y_val_one_hot, y_val) = split_train_val(
        X_train, y_train_one_hot, y_train, val_ratio=0.1
    )
    
    # Create model
    input_size = X_train.shape[1]
    output_size = y_train_one_hot.shape[1]
    hidden_sizes = [config.hidden_size] * config.num_layers
    
    model = NeuralNetwork(
        input_size,
        hidden_sizes,
        output_size,
        activation=config.activation,
        weight_init=config.weight_init
    )
    
    # Create loss function
    loss_fn = get_loss(config.loss)
    
    # Create optimizer
    optimizer_params = {
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay
    }
    
    if config.optimizer in ['momentum', 'nag']:
        optimizer_params['momentum'] = config.momentum
    elif config.optimizer == 'rmsprop':
        optimizer_params['beta'] = config.beta
        optimizer_params['epsilon'] = config.epsilon
    elif config.optimizer in ['adam', 'nadam']:
        optimizer_params['beta1'] = config.beta1
        optimizer_params['beta2'] = config.beta2
        optimizer_params['epsilon'] = config.epsilon
    
    optimizer = get_optimizer(config.optimizer, **optimizer_params)
    
    # Train model
    num_batches = int(np.ceil(X_train.shape[0] / config.batch_size))
    
    for epoch in range(config.epochs):
        # Training
        train_loss = 0
        train_acc = 0
        
        for X_batch, y_batch in get_batches(X_train, y_train_one_hot, config.batch_size):
            # Forward pass
            y_pred = model.forward(X_batch)
            
            # Compute loss
            loss = loss_fn.forward(y_pred, y_batch)
            train_loss += loss
            
            # Compute accuracy
            acc = accuracy(y_pred, y_batch)
            train_acc += acc
            
            # Backward pass
            grad_output = loss_fn.backward(y_pred, y_batch)
            model.backward(grad_output)
            
            # Update parameters
            optimizer.update(model.get_params_and_grads())
        
        train_loss /= num_batches
        train_acc /= num_batches
        
        # Validation
        y_val_pred = model.forward(X_val)
        val_loss = loss_fn.forward(y_val_pred, y_val_one_hot)
        val_acc = accuracy(y_val_pred, y_val_one_hot)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'loss': train_loss,
            'accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })
        
        print(f'Epoch {epoch+1}/{config.epochs} - '
              f'loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - '
              f'val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}')
    
    # Test model
    y_test_pred = model.forward(X_test)
    test_loss = loss_fn.forward(y_test_pred, y_test_one_hot)
    test_acc = accuracy(y_test_pred, y_test_one_hot)
    
    print(f'Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}')
    
    # Log test metrics
    wandb.log({
        'test_loss': test_loss,
        'test_accuracy': test_acc
    })
    
    # Plot confusion matrix
    y_test_pred_labels = np.argmax(y_test_pred, axis=1)
    
    class_names = None
    if config.dataset == 'fashion_mnist':
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    cm_path = plot_confusion_matrix(y_test, y_test_pred_labels, class_names)
    
    # Log confusion matrix
    wandb.log({'confusion_matrix': wandb.Image(cm_path)})
    
    # Plot examples
    if config.dataset == 'fashion_mnist' or config.dataset == 'mnist':
        # Plot 10 examples, 1 from each class
        fig, axs = plt.subplots(2, 5, figsize=(12, 6))
        axs = axs.flatten()
        
        for i in range(10):
            # Find an example of class i
            idx = np.where(y_test == i)[0][0]
            
            # Reshape image
            img = X_test[idx].reshape(28, 28)
            
            # Plot image
            axs[i].imshow(img, cmap='gray')
            axs[i].set_title(f'Class: {i}')
            axs[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('examples.png')
        plt.close()
        
        # Log examples
        wandb.log({'examples': wandb.Image('examples.png')})

def main():
    # Parse arguments
    args = parse_args()
    
    # Train model
    train_model(args)

if __name__ == '__main__':
    main()