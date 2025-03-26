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
from pynn import *

# Generate some data
def sine_data(samples=1000):
	X = np.arange(samples).reshape(-1, 1) / samples
	y = np.sin(2 * np.pi * X).reshape(-1, 1)
	return(X, y)
X, Y = sine_data()

# Define the network architecture
model = PyNN()
model.add(model.Dense(1, 64))
model.add(model.Sigmoid())
model.add(model.Dense(64, 64))
model.add(model.Sigmoid())
model.add(model.Dense(64, 1))
model.add(model.Linear())

# Show the network architecture
model.show()

# Train the model
model.train(
			X_train=None, Y_train=None,
			X_valid=None, Y_valid=None,
			X_tests=None, Y_tests=None,
			batch_size = None,
			loss='MSE',
			accuracy='regression',
			optimiser='SGD', lr=0.1, decay=0.0, beta1=0.9, beta2=0.999, e=1e-7,
			early_stop=False,
			epochs=100)
)

# Save the model
model.save()

# Load the model
model.load()

# Perform a prediction
prediction = model.predict(0.299)
print(prediction)
```

## Contributing
Feel free to contribute by submitting issues or pull requests. Ensure your code follows best practices and includes tests.
