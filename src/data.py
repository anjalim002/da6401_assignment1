import numpy as np
from tensorflow.keras.datasets import fashion_mnist, mnist

def load_data(dataset_name):
    """
    Load and preprocess MNIST or Fashion-MNIST dataset
    
    Args:
        dataset_name: String, either 'mnist' or 'fashion_mnist'
        
    Returns:
        Tuple of preprocessed training and testing data and labels
    """
    if dataset_name == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape from (N, 28, 28) to (N, 784)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # One-hot encode labels
    num_classes = 10
    y_train_one_hot = np.zeros((y_train.shape[0], num_classes))
    y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1
    
    y_test_one_hot = np.zeros((y_test.shape[0], num_classes))
    y_test_one_hot[np.arange(y_test.shape[0]), y_test] = 1
    
    return (X_train, y_train_one_hot, y_train), (X_test, y_test_one_hot, y_test)

def split_train_val(X_train, y_train_one_hot, y_train, val_ratio=0.1):
    """
    Split training data into training and validation sets
    
    Args:
        X_train: Training data
        y_train_one_hot: One-hot encoded training labels
        y_train: Training labels (not one-hot encoded)
        val_ratio: Ratio of validation data
        
    Returns:
        Tuple of training and validation data and labels
    """
    val_size = int(X_train.shape[0] * val_ratio)
    
    X_val = X_train[:val_size]
    y_val_one_hot = y_train_one_hot[:val_size]
    y_val = y_train[:val_size]
    
    X_train = X_train[val_size:]
    y_train_one_hot = y_train_one_hot[val_size:]
    y_train = y_train[val_size:]
    
    return (X_train, y_train_one_hot, y_train), (X_val, y_val_one_hot, y_val)

def get_batches(X, y, batch_size):
    """
    Generate batches from data
    
    Args:
        X: Input data
        y: Target data
        batch_size: Size of each batch
        
    Returns:
        Generator of batches
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield X[batch_indices], y[batch_indices]