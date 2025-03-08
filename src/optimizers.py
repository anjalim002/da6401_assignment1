import numpy as np

class Optimizer:
    """Base class for all optimizers"""
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
    def update(self, params_and_grads):
        raise NotImplementedError

class SGD(Optimizer):
    """Stochastic gradient descent optimizer"""
    def update(self, params_and_grads):
        """
        Update parameters using stochastic gradient descent
        
        Args:
            params_and_grads: List of (param, grad) tuples
        """
        for param, grad in params_and_grads:
            # Add L2 regularization
            param_grad = grad + self.weight_decay * param
            
            # Update parameters
            param -= self.learning_rate * param_grad

class Momentum(Optimizer):
    """Momentum optimizer"""
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.momentum = momentum
        self.velocity = {}
        
    def update(self, params_and_grads):
        """
        Update parameters using momentum
        
        Args:
            params_and_grads: List of (param, grad) tuples
        """
        for i, (param, grad) in enumerate(params_and_grads):
            # Initialize velocity if not exists
            if i not in self.velocity:
                self.velocity[i] = np.zeros_like(param)
                
            # Add L2 regularization
            param_grad = grad + self.weight_decay * param
            
            # Update velocity
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * param_grad
            
            # Update parameters
            param += self.velocity[i]

class NAG(Optimizer):
    """Nesterov accelerated gradient optimizer"""
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.momentum = momentum
        self.velocity = {}
        
    def update(self, params_and_grads):
        """
        Update parameters using Nesterov accelerated gradient
        
        Args:
            params_and_grads: List of (param, grad) tuples
        """
        for i, (param, grad) in enumerate(params_and_grads):
            # Initialize velocity if not exists
            if i not in self.velocity:
                self.velocity[i] = np.zeros_like(param)
                
            # Add L2 regularization
            param_grad = grad + self.weight_decay * param
            
            # Save previous velocity
            prev_velocity = self.velocity[i].copy()
            
            # Update velocity
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * param_grad
            
            # Update parameters with Nesterov correction
            param += -self.momentum * prev_velocity + (1 + self.momentum) * self.velocity[i]

class RMSprop(Optimizer):
    """RMSprop optimizer"""
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.epsilon = epsilon
        self.square_avg = {}
        
    def update(self, params_and_grads):
        """
        Update parameters using RMSprop
        
        Args:
            params_and_grads: List of (param, grad) tuples
        """
        for i, (param, grad) in enumerate(params_and_grads):
            # Initialize square average if not exists
            if i not in self.square_avg:
                self.square_avg[i] = np.zeros_like(param)
                
            # Add L2 regularization
            param_grad = grad + self.weight_decay * param
            
            # Update square average
            self.square_avg[i] = self.beta * self.square_avg[i] + (1 - self.beta) * param_grad**2
            
            # Update parameters
            param -= self.learning_rate * param_grad / (np.sqrt(self.square_avg[i]) + self.epsilon)

class Adam(Optimizer):
    """Adam optimizer"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
        
    def update(self, params_and_grads):
        """
        Update parameters using Adam
        
        Args:
            params_and_grads: List of (param, grad) tuples
        """
        self.t += 1
        
        for i, (param, grad) in enumerate(params_and_grads):
            # Initialize moments if not exists
            if i not in self.m:
                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)
                
            # Add L2 regularization
            param_grad = grad + self.weight_decay * param
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param_grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param_grad**2
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class NAdam(Optimizer):
    """NAdam optimizer (Adam with Nesterov momentum)"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
        
    def update(self, params_and_grads):
        """
        Update parameters using NAdam
        
        Args:
            params_and_grads: List of (param, grad) tuples
        """
        self.t += 1
        
        for i, (param, grad) in enumerate(params_and_grads):
            # Initialize moments if not exists
            if i not in self.m:
                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)
                
            # Add L2 regularization
            param_grad = grad + self.weight_decay * param
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param_grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param_grad**2
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Compute Nesterov momentum
            m_hat_nesterov = (self.beta1 * m_hat + (1 - self.beta1) * param_grad) / (1 - self.beta1**self.t)
            
            # Update parameters
            param -= self.learning_rate * m_hat_nesterov / (np.sqrt(v_hat) + self.epsilon)

def get_optimizer(name, **kwargs):
    """Factory function to get optimizer by name"""
    optimizers = {
        'sgd': SGD,
        'momentum': Momentum,
        'nag': NAG,
        'rmsprop': RMSprop,
        'adam': Adam,
        'nadam': NAdam
    }
    
    if name not in optimizers:
        raise ValueError(f"Optimizer {name} not supported")
        
    return optimizers[name](**kwargs)