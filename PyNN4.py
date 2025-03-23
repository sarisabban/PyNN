import math
import numpy as np
np.random.seed(42)

class PyNN():
	''' Lightweight NumPy-based neural network library '''
	def __init__(self):
		''' Initialise the class with the following '''
		self.layers = []
	def add(self, layer):
		''' Add an object to the network '''
		self.layers.append(layer)
	def cost(self, loss):
		''' The cost function '''
		return(np.mean(loss))
	def train(self, X, Y, loss_fn, optimiser, accuracy_fn, batch_size=None, epochs=1, verbose=1000):
		''' Train the network using forward and backward propagation '''
		loss = loss_fn()
		for epoch in range(epochs):
			output = X
			# Forward Propagation
			for layer in self.layers:
				output = layer.forward(output)
			y_pred = output
			cost = self.cost(loss.forward(Y, y_pred))
			acc = accuracy_fn(Y, y_pred)
			# Backpropagation
			grad = loss.backward(Y, y_pred)
			for i in range(len(self.layers) - 1, -1, -1):
				grad = self.layers[i].backward(grad)
			# Gradient Descent (only for Dense layers)
			for layer in self.layers:
				if isinstance(layer, Dense):
					optimiser(0.01, layer)
			# Print metrics
			if verbose and epoch % 1000 == 0:
				print(f"Epoch {epoch}: Loss={cost:.4f}, Accuracy={acc:.4f}")





def Parameters(inputs=1, outputs=1, alg='he uniform', sd=0.1, a=-0.5, b=0.5):
	''' Parameter Initialisation '''
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
		w =  np.random.normal(loc=0.0, scale=sd, size=(inputs, outputs))
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
	gamma = np.ones((1, outputs))
	beta = np.zeros((1, outputs))
	return(w, b, gamma, beta)

class Dense():
	''' A dense (fully connected) layer '''
	def __init__(self, inputs=1, outputs=1, alg='he uniform', sd=0.1, a=-0.5, b=0.5):
		''' Initialize weights and biases '''
		self.w, self.b, self.gamma, self.beta = Parameters(inputs, outputs, alg=alg, sd=sd, a=a, b=b)
		self.x = None
	def forward(self, x):
		''' Forward pass '''
		self.x = x
		return np.dot(x, self.w) + self.b
	def backward(self, grad):
		''' Backward pass '''
		self.dL_dw = np.dot(self.x.T, grad)
		self.dL_db = np.sum(grad, axis=0, keepdims=True)
		self.dL_dx = np.dot(grad, self.w.T)
		return(self.dL_dx)

class ReLU():
	''' The ReLU activation function '''
	def forward(self, z):
		''' The forward pass '''
		self.z = z
		y = np.maximum(0, self.z)
		return(y)
	def backward(self, DL_DY):
		''' The backward pass '''
		DA_Dz = DL_DY.copy()
		DA_Dz[self.z <= 0] = 0
		return(DA_Dz)

class Sigmoid():
	''' The Sigmoid activation function '''
	def forward(self, z):
		''' The forward pass '''
		self.y = 1 / (1 + np.exp(-z))
		return(self.y)
	def backward(self, DA_Dz):
		''' The backward pass '''
		DS_Dz = DA_Dz * (1 - self.y) * self.y
		return(DS_Dz)

class BCE_Loss():
	''' The Binary Cross-Entropy loss function '''
	def forward(self, y_true, y_pred):
		''' The forward pass '''
		y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
		loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
		return(loss)
	def backward(self, y_true,  y_pred):
		''' The backward pass '''
		y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
		DL_DY = (y_pred - y_true) / (y_pred * (1 - y_pred)) / len(y_pred)
		return(DL_DY)

def Binary_Accuracy(y_true, y_pred):
	predictions = (y_pred > 0.5) * 1
	accuracy = np.mean(predictions==y_true)
	return(accuracy)



def SGD(lr, layer):
	if hasattr(layer, 'dL_dw') and hasattr(layer, 'dL_db'):
		layer.w -= lr * layer.dL_dw
		layer.b -= lr * layer.dL_db


#def SGD(lr, w, b, dL_dw, dL_db):
#	w -= lr * dL_dw
#	b -= lr * dL_db
#	return(w, b)


















#----- Import Data -----#
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

X, Y = spiral_data(samples=100, classes=2) # X = (200, 2)   y = (200,)
Y = Y.reshape(-1, 1) # y = (200, 1)



model = PyNN()
model.add(Dense(2, 64, alg='random normal'))
model.add(ReLU())
model.add(Dense(64, 1, alg='random normal'))
model.add(Sigmoid())


#model.set(loss=BCE_Loss(), optimiser=SGD())

model.train(X, Y, BCE_Loss, SGD, Binary_Accuracy, epochs=200000)















