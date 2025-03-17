

print("Train.py has started execution for question 8!")

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
                        choices=['cross_entropy', 'mean_squared_error'],    
                        help='Loss function to use')
    
    # Optimizer arguments
    parser.add_argument('-o', '--optimizer', type=str, default='nadam',
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
    parser.add_argument('-nhl', '--num_layers', type=int, default=4,
                        help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', type=int, default=128,
                        help='Number of hidden neurons in a feedforward layer')
    parser.add_argument('-a', '--activation', type=str, default='sigmoid',
                        choices=['identity', 'sigmoid', 'tanh', 'ReLU'],
                        help='Activation function to use')
    
    # Added run tag for identifying runs in comparison
    parser.add_argument('-tag', '--run_tag', type=str, default='',
                        help='Tag to identify the run')
    
    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                        help='Run in comparison mode to compare different loss functions')
    
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
    cm_path = f'confusion_matrix_{wandb.run.name}.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm_path

def train_model(args):
    """
    Train neural network
    
    Args:
        args: Arguments
    
    Returns:
        Dictionary with training history
    """
    # Initialize wandb if not already initialized
    if wandb.run is None:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    
    config = wandb.config
    
    # Set run name for better tracking
    if wandb.run.name is None or wandb.run.name.startswith("sweep"):
        loss_name = config.loss
        run_tag = config.run_tag if hasattr(config, 'run_tag') else ""
        wandb.run.name = f"{config.optimizer}_hl{config.num_layers}_bs{config.batch_size}_act{config.activation}_{loss_name}{run_tag}"
    
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
    
    # Store training history
    history = {
        'epoch': [],
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
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
        
        # Store metrics in history
        history['epoch'].append(epoch)
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
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
        pred_path = f'prediction_examples_{wandb.run.name}.png'
        plt.savefig(pred_path, dpi=150)
        plt.close()
        
        # Log the predictions visualization
        wandb.log({'predictions': wandb.Image(pred_path)})
    
    # Call the prediction visualization function
    visualize_predictions()
    
    print(f"Training completed! Parameters used:")
    print(f"- Loss: {config.loss}")
    print(f"- Model: {config.num_layers} hidden layers with {config.hidden_size} neurons each")
    print(f"- Activation: {config.activation}")
    print(f"- Optimizer: {config.optimizer}")
    print(f"- Learning rate: {config.learning_rate}")
    print(f"- Weight initialization: {config.weight_init}")
    
    return history



def compare_losses():
    """
    Run and compare models with different loss functions
    """
    print("Starting loss function comparison...")
    
    # Create base arguments
    args = parse_args()
    
    # Use the best parameters from Book2.pdf
    args.optimizer = 'nadam'
    args.num_layers = 4
    args.batch_size = 64
    args.activation = 'sigmoid'
    args.learning_rate = 0.001
    args.hidden_size = 128
    args.weight_init = 'Xavier'
    args.epochs = 20
    args.beta1 = 0.9
    args.beta2 = 0.999
    args.epsilon = 1e-8
    args.weight_decay = 0
    args.dataset = 'fashion_mnist'
    
    loss_histories = {}
    loss_types = ['cross_entropy', 'mean_squared_error']   
    
    for loss_type in loss_types:
        print(f"\nTraining model with {loss_type} loss...")
        # Set the loss type for this run
        args.loss = loss_type
        args.run_tag = f"_{loss_type}"
        
        # Initialize a new wandb run
        with wandb.init(project=args.wandb_project, entity=args.wandb_entity, 
                        config=vars(args), name=f"{args.optimizer}_hl{args.num_layers}_bs{args.batch_size}_act{args.activation}_{loss_type}"):
            # Train the model
            history = train_model(args)
            loss_histories[loss_type] = history
    
    # Create a separate wandb run for comparison charts
    with wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name="Loss_Functions_Comparison", config=vars(args)):
        # Plot comparison charts and log to wandb
        plot_comparison_charts(loss_histories)
    
    print("Loss function comparison completed!")



def plot_comparison_charts(histories):
    """
    Plot charts comparing different loss functions and log them to wandb
    
    Args:
        histories: Dictionary with training histories for different loss types
    """
    # Set the plotting style
    plt.style.use('ggplot')
    
    # Create directory for comparison plots
    os.makedirs('media', exist_ok=True)
    
    # Set up figure and axes for all four plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Flatten axes for easier access
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # Common aspects of the plots
    colors = {'cross_entropy': 'blue', 'mean_squared_error': 'red'}
    legend_elements = []
    
    # List of metrics to plot
    metrics = [
        ('epoch', 'accuracy', 'Training Accuracy', ax1),
        ('epoch', 'val_accuracy', 'Validation Accuracy', ax2),
        ('epoch', 'loss', 'Training Loss', ax3),
        ('epoch', 'val_loss', 'Validation Loss', ax4)
    ]
    
    # Generate each plot
    for x_key, y_key, title, ax in metrics:
        for loss_type, history in histories.items():
            line, = ax.plot(
                history[x_key], 
                history[y_key], 
                color=colors[loss_type], 
                linewidth=2, 
                label=f"{loss_type}" if 'loss' in loss_type else f"{loss_type.replace('_', ' ').title()}"
            )
            if ax == ax1:  # Only add to legend once
                legend_elements.append(line)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add more ticks for better readability
        ax.set_xticks(range(0, 21, 5))
        
        # Format y-axis based on whether it's loss or accuracy
        if 'accuracy' in y_key:
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # Add common legend at the bottom
    fig.legend(
        handles=legend_elements, 
        labels=[label.get_label() for label in legend_elements],
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=12
    )
    
    # Overall figure title
    fig.suptitle('Comparison of Cross Entropy vs MSE Loss Functions', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the combined figure
    combined_path = 'media/combined_comparison.png'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    
    # Log the combined figure to wandb
    wandb.log({"combined_comparison": wandb.Image(combined_path)})
    
    # Individual charts
    individual_metrics = [
        ('epoch', 'accuracy', 'Training Accuracy', 'training_accuracy'),
        ('epoch', 'val_accuracy', 'Validation Accuracy', 'validation_accuracy'),
        ('epoch', 'loss', 'Training Loss', 'training_loss'),
        ('epoch', 'val_loss', 'Validation Loss', 'validation_loss')
    ]
    
    # Create a dictionary to store all individual plots for wandb
    wandb_images = {}
    
    for x_key, y_key, title, filename in individual_metrics:
        plt.figure(figsize=(8, 6))
        
        for loss_type, history in histories.items():
            plt.plot(
                history[x_key], 
                history[y_key], 
                color=colors[loss_type], 
                linewidth=2,
                label=f"{loss_type}" if 'loss' in loss_type else f"{loss_type.replace('_', ' ').title()}"
            )
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(title, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(range(0, 21, 5))
        
        if 'accuracy' in y_key:
            plt.ylim(0, 1.0)
            plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save the individual figure
        img_path = f'media/{filename}_comparison.png'
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        
        # Add to wandb images dictionary
        wandb_images[f"{filename}_comparison"] = wandb.Image(img_path)
        
        plt.close()
    
    # Log all individual images to wandb
    wandb.log(wandb_images)
    
    # Close the main figure
    plt.close(fig)
    
    print("Comparison charts saved locally and logged to wandb.")


def main():
    # Parse arguments
    args = parse_args()
    
    if args.compare:
        # Run comparison between MSE and Cross Entropy loss
        compare_losses()
    else:
        # Regular single training
        print(" Running single training with specified parameters...")
        print(" Using configuration:")
        print(f" - Loss: {args.loss}")
        print(f" - Epochs: {args.epochs}")
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




























































































