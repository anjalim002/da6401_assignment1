import numpy as np

class Layer:
    """Base class for all layers"""
    def __init__(self):
        self.params = {}
        self.grads = {}
        
    def forward(self, inputs):
        raise NotImplementedError
        
    def backward(self, grad_output):
        raise NotImplementedError

class Dense(Layer):
    """
    Fully connected layer implementing: output = input @ weights + bias
    """
    def __init__(self, input_size, output_size, weight_init='random'):
        super().__init__()
        
        # Initialize weights and bias
        # if weight_init == 'random':
        #     self.params['W'] = np.random.randn(input_size, output_size) * 0.01
        # elif weight_init == 'Xavier':
        #     # Xavier/Glorot initialization
        #     self.params['W'] = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
        # else:
        #     raise ValueError(f"Weight initialization {weight_init} not supported")
        if weight_init == 'He':
            self.params['W'] = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)  # ✅ He for ReLU
        elif weight_init == 'Xavier':
            self.params['W'] = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)  # ✅ Xavier
        else:
            self.params['W'] = np.random.randn(input_size, output_size) * 0.01  # Default (not recommended)    
        
        self.params['b'] = np.zeros(output_size)
        
        # Initialize gradients
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])
        
    def forward(self, inputs):
        """
        Forward pass
        
        Args:
            inputs: numpy array of shape (batch_size, input_size)
            
        Returns:
            outputs: numpy array of shape (batch_size, output_size)
        """
        self.inputs = inputs
        return np.dot(inputs, self.params['W']) + self.params['b']
        
    def backward(self, grad_output):
        """
        Backward pass
        
        Args:
            grad_output: numpy array of shape (batch_size, output_size)
            
        Returns:
            grad_input: numpy array of shape (batch_size, input_size)
        """
        # Calculate gradients
        self.grads['W'] = np.dot(self.inputs.T, grad_output)
        self.grads['b'] = np.sum(grad_output, axis=0)
        
        # Return gradient with respect to inputs
        return np.dot(grad_output, self.params['W'].T)