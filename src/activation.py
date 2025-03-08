import numpy as np

class Activation:
    """Base class for all activation functions"""
    def __init__(self):
        self.inputs = None
        
    def forward(self, inputs):
        self.inputs = inputs
        return self._forward(inputs)
    
    def backward(self, grad_output):
        return self._backward(grad_output)
    
    def _forward(self, inputs):
        raise NotImplementedError
        
    def _backward(self, grad_output):
        raise NotImplementedError

class Identity(Activation):
    """Identity activation function: f(x) = x"""
    def _forward(self, inputs):
        return inputs
        
    def _backward(self, grad_output):
        return grad_output

class Sigmoid(Activation):
    """Sigmoid activation function: f(x) = 1 / (1 + exp(-x))"""
    def _forward(self, inputs):
        self.outputs = 1.0 / (1.0 + np.exp(-inputs))
        return self.outputs
        
    def _backward(self, grad_output):
        return grad_output * self.outputs * (1 - self.outputs)

class Tanh(Activation):
    """Tanh activation function: f(x) = tanh(x)"""
    def _forward(self, inputs):
        self.outputs = np.tanh(inputs)
        return self.outputs
        
    def _backward(self, grad_output):
        return grad_output * (1 - self.outputs**2)

class ReLU(Activation):
    """ReLU activation function: f(x) = max(0, x)"""
    def _forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
        return self.outputs
        
    def _backward(self, grad_output):
        grad = np.array(grad_output, copy=True)
        grad[self.inputs <= 0] = 0
        return grad
class Softmax(Activation):
    """Softmax activation function: f(x) = exp(x) / sum(exp(x))"""
    def _forward(self, inputs):
        exps = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))  # Prevent overflow
        self.outputs = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.outputs

    def _backward(self, grad_output):
        """
        Compute gradient of Softmax activation.

        Softmax gradient is computed as:
        S_ij = softmax(output) * (delta_ij - softmax(output))

        Since Softmax is usually combined with CrossEntropyLoss,
        its derivative simplifies to (y_pred - y_true).
        """
        batch_size, num_classes = self.outputs.shape
        d_out = np.zeros((batch_size, num_classes, num_classes))

        for i in range(batch_size):
            softmax_out = self.outputs[i].reshape(-1, 1)
            d_out[i] = np.diagflat(softmax_out) - np.dot(softmax_out, softmax_out.T)

        return np.einsum("ijk,ik->ij", d_out, grad_output)  # Compute correct Softmax gradient



def get_activation(name):
    """Factory function to get activation by name"""
    activations = {
        'identity': Identity,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'ReLU': ReLU,
        'softmax' : Softmax
    }
    
    if name not in activations:
        raise ValueError(f"Activation {name} not supported")
        
    return activations[name]()