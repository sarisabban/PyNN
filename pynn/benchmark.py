import os
import sys
import sklearn
import zipfile
import numpy as np
from pynn import *

np.random.seed(42)

def sine_data(samples=1000):
	X = np.arange(samples).reshape(-1, 1) / samples
	Y = np.sin(2 * np.pi * X).reshape(-1, 1)
	return X, Y

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

def run_regression():
	''' Regression benchmark '''
	print('\nRunning Regression Example (Sine Wave)')
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

def run_binary_classification():
	''' Binary classification benchmark '''
	print('\nRunning Binary Classification Example')
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

def run_categorical_classification():
	''' Categorical classification benchmark '''
	print('\nRunning Categorical Classification Example')
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

def run_mnist():
	''' MNIST benchmark '''
	print('\nRunning MNIST Example')
	if not os.path.exists('mnist_train.csv'):
		print('[+] Downloading mnist_train.csv ...')
		os.system('wget -q https://github.com/phoebetronic/mnist/raw/refs/heads/main/mnist_train.csv.zip')
		with zipfile.ZipFile('mnist_train.csv.zip','r') as zip_ref: zip_ref.extractall('./')
		os.remove('mnist_train.csv.zip')

	if not os.path.exists('mnist_test.csv'):
		print('[+] Downloading mnist_test.csv ...')
		os.system('wget -q https://github.com/phoebetronic/mnist/raw/refs/heads/main/mnist_test.csv.zip')
		with zipfile.ZipFile('mnist_test.csv.zip','r') as zip_ref: zip_ref.extractall('./')
		os.remove('mnist_test.csv.zip')

	X_train = np.genfromtxt('./mnist_train.csv', delimiter=',')
	X_tests = np.genfromtxt('./mnist_test.csv', delimiter=',')

	Y_train = X_train[:,0]
	X_train = X_train[:,1:]
	Y_tests = X_tests[:,0]
	X_tests = X_tests[:,1:]

	X_train, Y_train = sklearn.utils.shuffle(X_train, Y_train)
	X_tests, Y_tests = sklearn.utils.shuffle(X_tests, Y_tests)

	X_train = X_train.astype('float32') / 255
	X_tests = X_tests.astype('float32') / 255

	Y_train = Y_train.astype(int)
	Y_tests = Y_tests.astype(int)

	Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
	Y_train = sklearn.preprocessing.OneHotEncoder().fit(Y_train).transform(Y_train).toarray()
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

def print_usage():
	print('Usage: python benchmark.py [example]')
	print('\nAvailable examples:')
	print('  regression   - Sine wave regression example')
	print('  binary       - Binary classification with spiral data')
	print('  categorical  - Categorical classification with spiral data')
	print('  mnist        - MNIST digit classification')
	print('  all          - Run all examples')

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print_usage()
		sys.exit(1)

	example = sys.argv[1].lower()

	if example == 'regression':
		run_regression()
	elif example == 'binary':
		run_binary_classification()
	elif example == 'categorical':
		run_categorical_classification()
	elif example == 'mnist':
		run_mnist()
	elif example == 'all':
		run_regression()
		run_binary_classification()
		run_categorical_classification()
		run_mnist()
	else:
		print(f"Error: Unknown example '{example}'")
		print_usage()
		sys.exit(1)
