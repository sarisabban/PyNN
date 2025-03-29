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
#----- Categorical-Classification Model -----#
np.random.seed(42)

def spiral_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    Y = np.zeros(samples*classes, dtype='uint8')
    for class_n in range(classes):
        ix = range(samples*class_n, samples*(class_n+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_n*4, (class_n+1)*4, samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        Y[ix] = class_n
    return X, Y

X, Y = spiral_data(samples=100, classes=3)

model = PyNN()
model.add(model.Dense(2, 64, alg='glorot uniform'))
model.add(model.LeakyReLU(alpha=0.05))
model.add(model.Dense(64, 3, alg='glorot uniform'))
model.add(model.Softmax())

model.show()

model.train(
	X_train=X, Y_train=Y,
	batch_size=None,
	loss='cce',
	accuracy='categorical',
	optimiser='adam', lr=0.005, decay=5e-7, beta1=0.9, beta2=0.999, e=1e-7,
	early_stop=False,
	epochs=2000,
	verbose=1)

#Train: epoch 2000/2000       Train Cost 0.16705 | Train Accuracy 0.94333 | 0s
```
```
#----- Binary-Classification Model -----#
np.random.seed(42)

def spiral_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    Y = np.zeros(samples*classes, dtype='uint8')
    for class_n in range(classes):
        ix = range(samples*class_n, samples*(class_n+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_n*4, (class_n+1)*4, samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        Y[ix] = class_n
    return X, Y

X, Y = spiral_data(samples=100, classes=2)
Y = Y.reshape(-1, 1)

model = PyNN()
model.add(model.Dense(2, 64))
model.add(model.ReLU())
model.add(model.Dense(64, 1))
model.add(model.Sigmoid())

model.show()

model.train(
	X_train=X, Y_train=Y,
	batch_size=None,
	loss='bce',
	accuracy='binary',
	optimiser='adam', lr=0.005, decay=5e-7, beta1=0.9, beta2=0.999, e=1e-7,
	early_stop=False,
	epochs=2000,
	verbose=1)

# Train: epoch 2000/2000       Train Cost 0.15278 | Train Accuracy 0.94500 | 0s
```
```
#----- Regression Model -----#
np.random.seed(42)

def sine_data(samples=1000):
	X = np.arange(samples).reshape(-1, 1) / samples
	y = np.sin(2 * np.pi * X).reshape(-1, 1)
	return(X, y)

X, Y = sine_data()

model = PyNN()
model.add(model.Dense(1, 64, alg='random normal'))
model.add(model.ReLU())
model.add(model.Dense(64, 64, alg='random normal'))
model.add(model.ReLU())
model.add(model.Dense(64, 1, alg='random normal'))
model.add(model.Linear())

model.show()

model.train(
	X_train=X, Y_train=Y,
	batch_size=None,
	loss='mse',
	accuracy='regression',
	optimiser='adam', lr=0.005, decay=1e-3, beta1=0.9, beta2=0.999, e=1e-7,
	early_stop=False,
	epochs=10000,
	verbose=1)

# Train: epoch 10000/10000     Train Cost 0.00000 | Train Accuracy 0.98200 | 0s
```
Saving the model and performing a prediction:
```py
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
