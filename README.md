# PyNN
Lightweight NumPy-based neural network library

## Overview
PyNN is a simple and lightweight neural network library built using NumPy. It provides the basic framework for constructing, training, and evaluating neural networks.

## Features
* Layers: Dense
* Activation function: Step, Linear, Sigmoid, ReLU, Leaky ReLU, TanH, Softmax
* Loss functions: MSE, MAE, BCE, CCE
* Accuracy functions: Regression, binary, categorical
* Optimisers: SGD, adagrad, RMSprop, adam
* Regularisation: L1L2, dropout, batch normalisation
* Weight initialisation algorithms:zeros, ones, random, glorot, he
* Saving/loading a model
* GPU acceleration

## Installation
```
pip install git+https://github.com/sarisabban/PyNN
```

## Usage
```py
import sklearn
from pynn import *

# Generate some data
def sine_data(samples=1000):
	X = np.arange(samples).reshape(-1, 1) / samples
	y = np.sin(2 * np.pi * X).reshape(-1, 1)
	return(X, y)
X, Y = sine_data()

X_train, X_valid, Y_train, Y_valid = sklearn.model_selection.train_test_split(X, Y, train_size=600)
X_valid, X_tests, Y_valid, Y_tests = sklearn.model_selection.train_test_split(X, Y, train_size=200)

# Define the network architecture
model = PyNN()
model.add(model.Dense(1, 64))
model.add(model.BatchNorm())
model.add(model.ReLU())
model.add(model.Dense(64, 64))
model.add(model.Sigmoid())
model.add(model.Dropout())
model.add(model.Dense(64, 1))
model.add(model.Linear())

# Show the network architecture
model.show()

# Train the model
model.train(X_train, Y_train,
			X_valid, Y_valid,
			X_tests, Y_tests,
			batch_size=32,
			loss='MSE',
			accuracy='regression',
			optimiser='SGD', lr=0.05, decay=0.2, beta1=0.9, beta2=0.999, e=1e-7,
			early_stop=False,
			epochs=100,
			verbose=1)

# Save the model
model.save('model.pkl')

# Load the model
model.load('model.pkl')

# Perform a prediction
prediction = model.predict(0.299)
print(prediction)
```

## Contributing
Feel free to contribute by submitting issues or pull requests. Ensure your code follows best practices and includes tests.
