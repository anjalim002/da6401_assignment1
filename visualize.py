import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Create a figure to display one example from each class
plt.figure(figsize=(12, 6))

# Find one example of each class
for i in range(10):
    # Get the first image of class i
    idx = np.where(y_train == i)[0][0]
    
    # Plot in a grid of 2x5
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[idx], cmap='gray')
    plt.title(class_names[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig('fashion_mnist_examples.png')
plt.show()

print("Examples saved to 'fashion_mnist_examples.png'")