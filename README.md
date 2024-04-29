# ML_Project15-FashionMNIST_DataAnalysis

### Fashion MNIST Classification with Convolutional Neural Network
This project explores building and training a Convolutional Neural Network (CNN) for classifying fashion items from the Fashion MNIST dataset.

### Data Acquisition and Preprocessing:

The Fashion MNIST dataset is downloaded using FashionMNIST.

Classes are identified: T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot.

Separate training and testing datasets are created.

Data augmentation is applied to the training data using random horizontal flips and rotations to increase the size and diversity of the training set.

Images are converted to tensors using ToTensor.

### Model Architecture:

A CNN architecture with convolutional and pooling layers is implemented.

The network uses ReLU activation and Batch Normalization for improved training stability.

The final layer outputs a prediction for each of the 10 fashion classes.

### Training and Evaluation:

A custom FMnist class encapsulates the model architecture, training, validation, and epoch-end logging functionalities.

Training is performed using the Adam optimizer and cross-entropy loss function.

The model is evaluated on the validation set after each epoch, tracking both loss and accuracy.

Training history is stored for visualization.

### Results:

The model achieves a validation accuracy of over 87% after 10 epochs, demonstrating its ability to learn and classify fashion items from the dataset.

Loss curves for training and validation data are plotted to visualize the training progress.

Accuracy curve on the validation set depicts the model's performance over epochs.

### Further Exploration:

Experiment with different hyperparameters (learning rate, number of epochs, network architecture) to potentially improve performance.

Explore more advanced data augmentation techniques like random cropping and color jittering.

Visualize the learned filters in the convolutional layers to understand what features the model focuses on for classification.

### Note:

The provided code includes helper functions for plotting loss and accuracy curves (plot_loss and plot_accs). You might need to adjust them based on your environment.
