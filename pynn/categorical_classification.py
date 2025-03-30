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

X, Y = spiral_data(samples=100, classes=3)

model = pynn.PyNN()
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
