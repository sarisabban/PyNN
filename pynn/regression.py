import numpy as np
from pynn import *

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
