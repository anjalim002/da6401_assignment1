import wandb
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# Initialize WandB
wandb.init(project="da6401_a1", name="Question_1")

# Load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Define class labels
class_labels = {
    0: 'T-shirt/Top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Boot'
}

# Create a subplot grid (2 rows, 5 columns)
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

# List for storing WandB images
wandb_images = []

for i in range(10):
    # Find the first image of the current class
    idx = np.where(y_train == i)[0][0]
    image = x_train[idx]

    # Plot the image in VS Code
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(class_labels[i])
    axes[i].axis('off')

    # Append image to WandB log
    wandb_images.append(wandb.Image(image, caption=class_labels[i]))

# Show the plot in VS Code terminal
plt.tight_layout()
plt.show()

# Log images to WandB
wandb.log({"Fashion-MNIST Samples": wandb_images})

# Finish WandB run
wandb.finish()
