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
```
# ----- MNIST -----#
import sklearn

np.random.seed(42)
# Download the dataset https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
# unzip the dataset

X_train = np.genfromtxt('/home/slurm/Desktop/mnist_train.csv', delimiter=',')
X_tests = np.genfromtxt('/home/slurm/Desktop/mnist_test.csv', delimiter=',')

Y_train = X_train[:,0 ] # (60000, 784)
X_train = X_train[:,1:] # (60000,)
Y_tests = X_tests[:,0 ] # (10000, 784)
X_tests = X_tests[:,1:] # (10000,)

# Shuffle datasets
X_train, Y_train = sklearn.utils.shuffle(X_train, Y_train)
X_tests, Y_tests = sklearn.utils.shuffle(X_tests, Y_tests)

X_train, X_valid, Y_train, Y_valid = sklearn.model_selection.train_test_split(X_train, Y_train, train_size=50000)

# Scale images to a [0, 1] range
X_train = X_train.astype("float32") / 255
X_valid = X_tests.astype("float32") / 255
X_tests = X_tests.astype("float32") / 255

# Ensure Y labels are integers
Y_train = Y_train.astype(int)
Y_valid = Y_valid.astype(int)
Y_tests = Y_tests.astype(int)

# One-hot encoding
Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
Y_train = sklearn.preprocessing.OneHotEncoder().fit(Y_train).transform(Y_train).toarray()
Y_valid = np.reshape(Y_valid, (Y_valid.shape[0], 1))
Y_valid = sklearn.preprocessing.OneHotEncoder().fit(Y_valid).transform(Y_valid).toarray()
Y_tests = np.reshape(Y_tests, (Y_tests.shape[0], 1))
Y_tests = sklearn.preprocessing.OneHotEncoder().fit(Y_tests).transform(Y_tests).toarray()

model = PyNN()
model.add(model.Dense(784, 256))
model.add(model.ReLU())
model.add(model.Dropout(0.45))
model.add(model.Dense(256, 256))
model.add(model.ReLU())
model.add(model.Dropout(0.45))
model.add(model.Dense(256, 10))
model.add(model.Softmax())

model.show()

model.train(
	X_train=X_train, Y_train=Y_train,
#	X_valid=X_valid, Y_valid=Y_valid,
	X_tests=X_tests, Y_tests=Y_tests,
	batch_size=128,
	loss='cce',
	accuracy='categorical',
	optimiser='adam', lr=1e-4, decay=5e-7, beta1=0.9, beta2=0.999, e=1e-7,
	early_stop=False,
	epochs=20,
	verbose=2)
```

The training output
```
PyNN is running on CPU
------------------------------------------------------------
Layer                    Shape                    Parameters
------------------------------------------------------------
Dense                    (784, 256)               200704
ReLU                                              
Dropout                                           
Dense                    (256, 256)               65536
ReLU                                              
Dropout                                           
Dense                    (256, 10)                2560
Softmax                                           
------------------------------
Total Parameters: 268,800

Train: epoch 1/20            Train Cost 0.48725 | Train Accuracy 0.87276 | 64s
Train: epoch 2/20            Train Cost 0.39401 | Train Accuracy 0.89286 | 65s
Train: epoch 3/20            Train Cost 0.35661 | Train Accuracy 0.90144 | 64s
Train: epoch 4/20            Train Cost 0.33407 | Train Accuracy 0.90622 | 64s
Train: epoch 5/20            Train Cost 0.31704 | Train Accuracy 0.91050 | 63s
Train: epoch 6/20            Train Cost 0.30317 | Train Accuracy 0.91410 | 73s
Train: epoch 7/20            Train Cost 0.29148 | Train Accuracy 0.91668 | 75s
Train: epoch 8/20            Train Cost 0.28183 | Train Accuracy 0.91930 | 64s
Train: epoch 9/20            Train Cost 0.27278 | Train Accuracy 0.92144 | 66s
Train: epoch 10/20           Train Cost 0.26523 | Train Accuracy 0.92314 | 65s
Train: epoch 11/20           Train Cost 0.25826 | Train Accuracy 0.92512 | 64s
Train: epoch 12/20           Train Cost 0.25108 | Train Accuracy 0.92710 | 66s
Train: epoch 13/20           Train Cost 0.24474 | Train Accuracy 0.92898 | 64s
Train: epoch 14/20           Train Cost 0.23927 | Train Accuracy 0.93002 | 64s
Train: epoch 15/20           Train Cost 0.23328 | Train Accuracy 0.93200 | 67s
Train: epoch 16/20           Train Cost 0.22803 | Train Accuracy 0.93334 | 68s
Train: epoch 17/20           Train Cost 0.22293 | Train Accuracy 0.93480 | 62s
Train: epoch 18/20           Train Cost 0.21819 | Train Accuracy 0.93604 | 65s
Train: epoch 19/20           Train Cost 0.21306 | Train Accuracy 0.93750 | 64s
Train: epoch 20/20           Train Cost 0.20835 | Train Accuracy 0.93862 | 63s
Tests:                       Tests Cost 0.20469 | Tests Accuracy 0.93840 | 1s
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
