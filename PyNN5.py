import math
import numpy as np
np.random.seed(42)  # Ensure reproducibility

class PyNN():
	def __init__(self):
		self.layers = []
	def add(self, layer):
		self.layers.append(layer)
	def cost(self, loss):
		return np.mean(loss)
	def train(self, X, Y, loss_fn, optimiser, accuracy_fn, batch_size=32, epochs=20000, verbose=1000, patience=10):
		loss = loss_fn()
		best_loss = float('inf')
		patience_counter = 0

		for epoch in range(epochs):
#			indices = np.random.permutation(len(X))
#			X_shuffled, Y_shuffled = X[indices], Y[indices]

#			for i in range(0, len(X), batch_size):
#				X_batch = X_shuffled[i:i+batch_size]
#				Y_batch = Y_shuffled[i:i+batch_size]
                
			output = X#X_batch
			y_true = Y#Y_batch
			for layer in self.layers:
				output = layer.forward(output)
			y_pred = output

			cost = self.cost(loss.forward(y_true, y_pred))
			acc = accuracy_fn(y_true, y_pred)

			grad = loss.backward(y_true, y_pred)
			for i in range(len(self.layers) - 1, -1, -1):
				grad = self.layers[i].backward(grad)

			for layer in self.layers:
				if isinstance(layer, Dense):
					optimiser(0.01, layer)

#			if verbose and epoch % verbose == 0:
			print(f"Epoch {epoch}: Loss={cost:.4f}, Accuracy={acc:.4f}")

#			if cost < best_loss:
#				best_loss = cost
#				patience_counter = 0
#			else:
#				patience_counter += 1
#				if patience_counter >= patience:
#					print(f"Early stopping at epoch {epoch}")
#					return


def Parameters(inputs=1, outputs=1, alg='he uniform', sd=0.1, a=-0.5, b=0.5):
    if alg == 'zeros':
        w = np.zeros((inputs, outputs))
    elif alg == 'ones':
        w = np.ones((inputs, outputs))
    elif alg == 'random normal':
        w = np.random.normal(loc=0.0, scale=sd, size=(inputs, outputs))
    elif alg == 'random uniform':
        w = np.random.uniform(low=a, high=b, size=(inputs, outputs))
    elif alg == 'glorot normal':
        sd = 1 / (math.sqrt(inputs + outputs))
        w = np.random.normal(loc=0.0, scale=sd, size=(inputs, outputs))
    elif alg == 'glorot uniform':
        a = - math.sqrt(6) / (math.sqrt(inputs + outputs))
        b = math.sqrt(6) / (math.sqrt(inputs + outputs))
        w = np.random.uniform(low=a, high=b, size=(inputs, outputs))
    elif alg == 'he normal':
        sd = 2 / (math.sqrt(inputs))
        w = np.random.normal(loc=0.0, scale=sd, size=(inputs, outputs))
    elif alg == 'he uniform':
        a = - math.sqrt(6) / (math.sqrt(inputs))
        b = math.sqrt(6) / (math.sqrt(inputs))
        w = np.random.uniform(low=a, high=b, size=(inputs, outputs))
    b = np.zeros((1, outputs))
    return w, b

class Dense():
    def __init__(self, inputs=1, outputs=1, alg='he uniform', sd=0.1, a=-0.5, b=0.5):
        self.w, self.b = Parameters(inputs, outputs, alg=alg, sd=sd, a=a, b=b)
        self.x = None
    
    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b
    
    def backward(self, grad):
        self.dL_dw = np.dot(self.x.T, grad)
        self.dL_db = np.sum(grad, axis=0, keepdims=True)
        self.dL_dx = np.dot(grad, self.w.T)
        return self.dL_dx

class ReLU():
    def forward(self, z):
        self.z = z
        return np.maximum(0, self.z)
    
    def backward(self, DL_DY):
        DA_Dz = DL_DY.copy()
        DA_Dz[self.z <= 0] = 0
        return DA_Dz

class Sigmoid():
    def forward(self, z):
        self.y = 1 / (1 + np.exp(-z))
        return self.y
    
    def backward(self, DA_Dz):
        return DA_Dz * (1 - self.y) * self.y

class BCE_Loss():
    def forward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / len(y_pred)

def Binary_Accuracy(y_true, y_pred):
    predictions = (y_pred > 0.5) * 1
    return np.mean(predictions == y_true)

def SGD(lr, layer):
    if hasattr(layer, 'dL_dw') and hasattr(layer, 'dL_db'):
        layer.w -= lr * layer.dL_dw
        layer.b -= lr * layer.dL_db

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
model.add(Dense(2, 64, alg='random normal'))
model.add(ReLU())
model.add(Dense(64, 1, alg='random normal'))
model.add(Sigmoid())

model.train(X, Y, BCE_Loss, SGD, Binary_Accuracy, epochs=200000)

