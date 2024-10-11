# Neural-Netwrok-Models
Understanding some most commonly used neural networks
1. MLP
2. CNN
3. LSTM

## MLUs
- The first layer specifies input_shape=(10,), indicating that it expects input data with 10 features. It has 64 units/neurons with an activation function: **ReLU (Rectified Linear Unit)**. ReLU is commonly used in hidden layers to introduce non-linearity.
- The second layer has 32 units/neurons with an activation function: ReLU. The output layer has 1 unit because it’s a binary classification problem with an activation function: **Sigmoid**. The sigmoid function squashes the output between 0 and 1, which is suitable for binary classification problems.

## CNNs
- The first layer is a convolutional layer specified by the **Conv2D** function. It applies a set of filters to the input data to detect features. **32** represents the number of filters, and **(3, 3)** signifies the filter size. We have used an activation function: **ReLU (Rectified Linear Unit)**. ReLU introduces non-linearity to the model, aiding in learning complex patterns in the data. Input shape: **(28, 28, 3)** indicates that the input images are 28×28 pixels with 3 channels (RGB).
- The **pooling layer** is used to reduce the spatial dimensions of the feature maps obtained from the convolutional layers, thus decreasing the computational complexity. Here **(2, 2)** represents the pool size for max pooling operation, which reduces each spatial dimension by half.
- The **flatten layer** is used to convert the multi-dimensional feature maps into a one-dimensional vector, preparing the data for the fully connected layers. It is necessary because dense layers require one-dimensional input.
- After flattening, there are two **dense layers**. The first dense layer has 64 units with ReLU activation, which allows the model to learn complex patterns in the flattened feature vectors. The second dense layer has 10 units, which corresponds to the number of classes in the classification task. We have used an activation function: **Softmax**. The softmax activation function produces a probability distribution over the classes, making it suitable for multi-class classification.

## LSTMs
- The LSTM layer is a recurrent neural network (RNN) layer with memory cells that allow the model to retain information over time. **64** represents the number of memory units or neurons in the LSTM layer. **input_shape=(10, 1)** specifies the shape of the input data. Here, 10 is the sequence length, and 1 is the number of features per time step. Input data is expected to be in the shape (batch_size, sequence_length, num_features).
- Following the LSTM layer, there is a dense layer with a single neuron with an activation function: **Sigmoid**. Sigmoid activation is commonly used in binary classification tasks to produce probabilities.
