import numpy as np
from src.activation import get_activation
from src.layers import Dense

class NeuralNetwork:
    """
    Feedforward neural network
    """
    def __init__(self, input_size, hidden_sizes, output_size, activation='sigmoid', weight_init='random'):
        """
        Initialize neural network
        
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Size of output features
            activation: Activation function name
            weight_init: Weight initialization method
        """
        self.layers = []
        self.activations = []
        
        # Add input layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Create layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i+1], weight_init=weight_init))
            
            # Use sigmoid activation for output layer, otherwise use specified activation
            if i == len(layer_sizes) - 2:
                self.activations.append(get_activation('softmax'))
            else:
                self.activations.append(get_activation(activation))
    
    def forward(self, inputs):
        """
        Forward pass
        
        Args:
            inputs: numpy array of shape (batch_size, input_size)
            
        Returns:
            outputs: numpy array of shape (batch_size, output_size)
        """
        outputs = inputs
        
        for layer, activation in zip(self.layers, self.activations):
            outputs = layer.forward(outputs)
            outputs = activation.forward(outputs)
            
        return outputs
    
    def backward(self, grad_output):
        """
        Backward pass
        
        Args:
            grad_output: numpy array of shape (batch_size, output_size)
            
        Returns:
            grad_input: numpy array of shape (batch_size, input_size)
        """
        grad = grad_output
        
        for layer, activation in reversed(list(zip(self.layers, self.activations))):
            grad = activation.backward(grad)
            grad = layer.backward(grad)
            
        return grad
    
    def get_params(self):
        """
        Get all parameters
        
        Returns:
            params: List of parameters
        """
        params = []
        
        for layer in self.layers:
            params.append(layer.params)
            
        return params
    
    def get_grads(self):
        """
        Get all gradients
        
        Returns:
            grads: List of gradients
        """
        grads = []
        
        for layer in self.layers:
            grads.append(layer.grads)
            
        return grads
    
    def get_params_and_grads(self):
        """
        Get all parameters and gradients
        
        Returns:
            params_and_grads: List of (param, grad) tuples
        """
        params_and_grads = []
        
        for layer in self.layers:
            for param_name in layer.params:
                params_and_grads.append((
                    layer.params[param_name], 
                    layer.grads[param_name]
                ))
                
        return params_and_grads