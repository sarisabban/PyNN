import sklearn
import numpy as np
from pynn import *

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
	X_tests=X_tests, Y_tests=Y_tests,
	batch_size=128,
	loss='cce',
	accuracy='categorical',
	optimiser='adam', lr=1e-4, decay=5e-7, beta1=0.9, beta2=0.999, e=1e-7,
	early_stop=False,
	epochs=20,
	verbose=2)

# 
