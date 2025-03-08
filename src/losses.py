import numpy as np

class Loss:
    """Base class for all loss functions"""
    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)
        
    def forward(self, predictions, targets):
        raise NotImplementedError
        
    def backward(self, predictions, targets):
        raise NotImplementedError

class CrossEntropyLoss(Loss):
    """Cross entropy loss function"""
    def forward(self, predictions, targets):
        """
        Compute cross entropy loss
        
        Args:
            predictions: numpy array of shape (batch_size, num_classes)
            targets: numpy array of shape (batch_size, num_classes)
            
        Returns:
            loss: scalar
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        
        # Clip predictions to avoid numerical instability
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Compute cross entropy loss
        batch_size = predictions.shape[0]
        loss = -np.sum(targets * np.log(predictions)) / batch_size
        
        return loss
        
    def backward(self, predictions, targets):
        """
        Compute gradient of cross entropy loss
        
        Args:
            predictions: numpy array of shape (batch_size, num_classes)
            targets: numpy array of shape (batch_size, num_classes)
            
        Returns:
            gradient: numpy array of shape (batch_size, num_classes)
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        
        # Clip predictions to avoid numerical instability
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Compute gradient
        batch_size = predictions.shape[0]
        # gradient = -targets / predictions / batch_size
        
        return (predictions - targets) / batch_size

class MeanSquaredErrorLoss(Loss):
    """Mean squared error loss function"""
    def forward(self, predictions, targets):
        """
        Compute mean squared error loss
        
        Args:
            predictions: numpy array of shape (batch_size, num_classes)
            targets: numpy array of shape (batch_size, num_classes)
            
        Returns:
            loss: scalar
        """
        batch_size = predictions.shape[0]
        loss = np.sum((predictions - targets) ** 2) / (2 * batch_size)
        
        return loss
        
    def backward(self, predictions, targets):
        """
        Compute gradient of mean squared error loss
        
        Args:
            predictions: numpy array of shape (batch_size, num_classes)
            targets: numpy array of shape (batch_size, num_classes)
            
        Returns:
            gradient: numpy array of shape (batch_size, num_classes)
        """
        batch_size = predictions.shape[0]
        gradient = (predictions - targets) / batch_size
        
        return gradient

def get_loss(name):
    """Factory function to get loss by name"""
    losses = {
        'cross_entropy': CrossEntropyLoss,
        'mean_squared_error': MeanSquaredErrorLoss
    }
    
    if name not in losses:
        raise ValueError(f"Loss {name} not supported")
        
    return losses[name]()