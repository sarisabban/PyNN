import numpy as np
from pynn import *

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
