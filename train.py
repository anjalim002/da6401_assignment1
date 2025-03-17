print("Train.py has started execution!")

import argparse
import wandb
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import math

print("Imports successful")

from src.data import load_data, split_train_val, get_batches
from src.network import NeuralNetwork
from src.losses import get_loss
from src.optimizers import get_optimizer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network')
    
    # Wandb arguments
    parser.add_argument('-wp', '--wandb_project', type=str, default='da6401_a1',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default='da24m002-indian-institute-of-technology-madras',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    
    # Dataset arguments
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',
                        choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use')

    # Training arguments
    parser.add_argument('-e', '--epochs', type=int, default=20,  
                        help='Number of epochs to train neural network')
    parser.add_argument('-b', '--batch_size', type=int, default=64,  
                        help='Batch size used to train neural network')
    

    # Loss arguments
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy'],    
                        help='Loss function to use')
    
    # Optimizer arguments
    parser.add_argument('-o', '--optimizer', type=str, default='nadam',  # Changed from adam to nadam based on best params
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimizer to use')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,  # Keeping 0.001 as it's already the best param
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
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0,  # Keeping 0.0 as it's the best param
                        help='Weight decay used by optimizers')
    
    # Model arguments
    parser.add_argument('-w_i', '--weight_init', type=str, default='Xavier',  # Keeping Xavier as it's already the best param
                        choices=['random', 'Xavier'],
                        help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=4,  # Changed from 3 to 4 based on best params
                        help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', type=int, default=128,  # Keeping 128 as it's already the best param
                        help='Number of hidden neurons in a feedforward layer')
    parser.add_argument('-a', '--activation', type=str, default='sigmoid',  # Changed from ReLU to sigmoid based on best params
                        choices=['identity', 'sigmoid', 'tanh', 'ReLU'],
                        help='Activation function to use')  
    # Sweep flag
    parser.add_argument('--sweep', action='store_true',
                        help='Run as part of a sweep')
    
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
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Path to saved confusion matrix image
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Set non-interactive backend to avoid display requirements
    plt.switch_backend('agg')
    
    # Enhanced plot with better visual elements
    plt.figure(figsize=(16,14))
    
    # Use a better colormap and improve visualization
    ax = sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto",
                cbar_kws={'label': 'Normalized Frequency'},
                linewidths=0.5, linecolor='grey')
    
    # Rotate tick labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add labels and title with better fonts
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Fashion MNIST', fontsize=14, fontweight='bold')
    
    # Tight layout to ensure everything fits
    plt.tight_layout()
    
    # Save confusion matrix to file with higher DPI for better quality
    cm_path = 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm_path

def train_model(args):
    """
    Train neural network
    
    Args:
        args: Arguments
    """
    # Initialize wandb if not already initialized
    if wandb.run is None:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    
    config = wandb.config
    
    # Set run name for better tracking
    if wandb.run.name is None or wandb.run.name.startswith("sweep"):
        wandb.run.name = f"{config.optimizer}_hl{config.num_layers}_bs{config.batch_size}_act{config.activation}"
    
    print(f" Starting training with config: {config}")
    
    # Load data
    (X_train, y_train_one_hot, y_train), (X_test, y_test_one_hot, y_test) = load_data(config.dataset)
    
    # Split training data into training and validation sets
    (X_train, y_train_one_hot, y_train), (X_val, y_val_one_hot, y_val) = split_train_val(
        X_train, y_train_one_hot, y_train, val_ratio=0.1
    )
    
    # Create model
    input_size = X_train.shape[1]
    output_size = y_train_one_hot.shape[1]
    
    # Create model
    input_size = X_train.shape[1]
    output_size = y_train_one_hot.shape[1]

    # Use uniform hidden sizes
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
    
    # Modified to run only 1 epoch instead of config.epochs
    for epoch in range(config.epochs):  # Use full number of epochs from config
        
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
        
        print(f'Epoch {epoch+1}/{config.epochs} - '  # Changed from config.epochs to 1
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
    
    def visualize_predictions():
        # Get random samples from test set
        num_samples = 10
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        # Make predictions
        samples = X_test[indices]
        predictions = model.forward(samples)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = y_test[indices]
        
        # Plot
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        axs = axs.flatten()
        
        for i in range(num_samples):
            # Reshape image for display
            img = samples[i].reshape(28, 28)
            
            # Get class names
            if config.dataset == 'fashion_mnist':
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                true_name = class_names[true_labels[i]]
                pred_name = class_names[pred_labels[i]]
            else:
                true_name = str(true_labels[i])
                pred_name = str(pred_labels[i])
            
            # Color based on correctness
            title_color = 'green' if true_labels[i] == pred_labels[i] else 'red'
            
            # Plot
            axs[i].imshow(img, cmap='gray')
            axs[i].set_title(f'True: {true_name}\nPred: {pred_name}', 
                           color=title_color, fontsize=9)
            axs[i].axis('off')
        
        plt.tight_layout()
        pred_path = 'prediction_examples.png'
        plt.savefig(pred_path, dpi=150)
        plt.close()
        
        # Log the predictions visualization
        wandb.log({'predictions': wandb.Image(pred_path)})
    
    # Call the prediction visualization function
    visualize_predictions()
    
    print(f"Training completed! Best parameters used:")
    print(f"- Model: {config.num_layers} hidden layers with {config.hidden_size} neurons each")
    print(f"- Activation: {config.activation}")
    print(f"- Optimizer: {config.optimizer}")
    print(f"- Learning rate: {config.learning_rate}")
    print(f"- Weight initialization: {config.weight_init}")
    print(f"- Confusion matrix saved to {cm_path}")
    
    return test_acc

def sweep_train():
    """
    Training function for WandB sweep
    """
    # Initialize wandb with sweep config
    run = wandb.init()
    
    # Create argument object with default values
    args = parse_args()
    
    # Override arguments with sweep config
    for key, value in wandb.config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    
    # Set custom run name
    # run.name = f"{args.optimizer}_hl{args.num_layers}_bs{args.batch_size}_act{args.activation}"
    wandb.run.name = f"{args.optimizer}_hl{args.num_layers}_bs{args.batch_size}_act{args.activation}"
    wandb.run.save()

    # Add default optimizer parameters if not present in the sweep config
    # This is the key fix - we need to manually ensure these parameters exist
    if args.optimizer in ['momentum', 'nag'] and 'momentum' not in wandb.config:
        wandb.config.update({'momentum': args.momentum}, allow_val_change=True)
    
    if args.optimizer == 'rmsprop' and 'beta' not in wandb.config:
        wandb.config.update({'beta': args.beta}, allow_val_change=True)
    
    if args.optimizer in ['adam', 'nadam']:
        if 'beta1' not in wandb.config:
            wandb.config.update({'beta1': args.beta1}, allow_val_change=True)
        if 'beta2' not in wandb.config:
            wandb.config.update({'beta2': args.beta2}, allow_val_change=True)
    
    if (args.optimizer in ['momentum', 'nag', 'rmsprop', 'adam', 'nadam'] and 
            'epsilon' not in wandb.config):
        wandb.config.update({'epsilon': args.epsilon}, allow_val_change=True)
   
    # Train model with sweep config
    train_model(args)

def main():
    # Parse arguments
    args = parse_args()
    
    
    args.sweep = False  # Set to False to run with best parameters
    
    # Define sweep configuration - commented out as we're not using it now
    """
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'epochs': {
                'values': [10,15,20]
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            'num_layers': {
                'values': [3, 4, 5]
            },
            'hidden_size': {
                'values': [32, 64, 128]
            },
            'learning_rate': {
                'values': [0.001, 0.0001]
            },
            'weight_decay': {
                'values': [0, 0.0005, 0.5]
            },
            'optimizer': {
                'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
            },
            'weight_init': {
                'values': ['random', 'Xavier']
            },
            'activation': {
                'values': ['sigmoid', 'tanh', 'ReLU']
            },

            'loss': {
                'values': ['cross_entropy']
                
            },
            'dataset': {
                'value': 'fashion_mnist'
            },
            # Add optimizer-specific parameters
            'momentum': {
                'value': 0.9  # Used by momentum and nag optimizers
            },
            'beta': {
                'value': 0.9  # Used by rmsprop optimizer
            },
            'beta1': {
                'value': 0.9  # Used by adam and nadam optimizers
            },
            'beta2': {
                'value': 0.999  # Used by adam and nadam optimizers
            },
            'epsilon': {
                'value': 1e-8  # Used by multiple optimizers
            }
        }
    }
    """
    
    if args.sweep:
        print(" Running sweep...")
        # Run sweep
        # Comment out sweep execution
        """
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
        print(f" Sweep created with ID: {sweep_id}")
        print(" Starting sweep agent...")
        wandb.agent(sweep_id, function=sweep_train, count=3)
        """
        print(" Sweep mode is enabled but we're skipping it.")
    else:
        # Train model with the best parameters
        print(" Running single training with best parameters from previous sweep...")
        print(" Using configuration:")
        print(f" - Epochs: 20 ")
        print(f" - Batch size: {args.batch_size}")
        print(f" - Hidden layers: {args.num_layers}")
        print(f" - Hidden size: {args.hidden_size}")
        print(f" - Activation: {args.activation}")
        print(f" - Optimizer: {args.optimizer}")
        print(f" - Learning rate: {args.learning_rate}")
        print(f" - Weight initialization: {args.weight_init}")
        train_model(args)

if __name__ == '__main__':
    main()