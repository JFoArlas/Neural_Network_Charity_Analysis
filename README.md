# Neural Network Charity Analysis

## Overview
The purpose of this analysis was to help a foundation predict where to make investments by using machine learning and neural networks. To do this, we were given a CSV containing more than 34,000 organizations that have received funding over the years. We were then tasked with creating a binary classifier capable of predicting whether applicants will be successful if funded.

## Results

- Data Preprocessing
  - The `IS_SUCCESSFUL` column is considered the target for the model.
  - The `EIN` and `NAME` columns are neither targets nor features, and were therefore removed from the input data.
  - The remaining columns are considered to be the features for the model.

- Compiling, Training, and Evaluating the Model
  - My original model was made up of 2 hidden layers, one with 80 neurons and one with 30 neurons. I used the ReLU activation function for the hidden layers and the Sigmoid activation function for the output layer:


  
  ```
  # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
  number_input_features = len(X_train_scaled[0])
  hidden_nodes_layer1 = 80
  hidden_nodes_layer2 = 30

  nn = tf.keras.models.Sequential()

  # First hidden layer
  nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

  # Second hidden layer
  nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

  # Output layer
  nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
  ```
  
  - I was not able to achieve the target model performance.
  - To try and increase model performance I adjusted the number of epochs, added an additional hidden layer, and adjusted the number of neurons in each layer.

## Summary
The original model resulted in **72.39% accuracy**.

1. In my first attempt to optimize the model I added a third hidden layer with 30 nodes, in addition to increasing the nodes in layer 1 from 80 to 100, and increasing the nodes in layer 2 from 30 to 50. Lastly I tried increasing the epochs to 150. All of this resulted in **72.63% accuracy**, which was just slightly higher than the original model.

2. In my second attempt to optimize the model I kept three hidden layers and adjusted the nodes to 80, 30, and 10 respectively. I also decreased the epochs back to 100 and changed the activation function for the hidden layers from ReLU to tanh. All of this resulted in **72.57% accuracy**, which was just slightly lower than the first attempt but still a bit higher than original model.

3. In my third attempt to optimize the model I again kept the three hidden layers and adjusted the nodes in the first layer to 50, keeping 30 and 10 nodes in the second and third layers respectively. I also kept 100 epochs and changed the activation function for the hidden layers back to ReLU since the first attempt was more successful with that than with tanh. All of this resulted in **74.17% accuracy**, which was the highest I was able to achieve.
