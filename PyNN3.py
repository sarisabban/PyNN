import math
import numpy as np
np.random.seed(42)

class PyNN():
	''' Lightweight NumPy-based neural network library '''
	#---------- Utilities ---------- #
	def __init__(self):
		''' Initialise the class with the following '''
		self.layers = []
	def add(self, layer):
		''' Add an object to the network '''
		self.layers.append(layer)
	def cost(self, loss):
		''' The cost function '''
		return(np.mean(loss))
	def verbosity(self, epoch, cost, accuracy, *, verbose=1):
		''' Control level of information printout during training '''
		E = epoch
		C = cost
		A = accuracy
		string = f'Epoch: {E:,} | Cost: {C:.5f} | Accuracy: {A:.5f}'
		if verbose == 0:
			pass
		elif verbose == 1:
			if epoch % 100 == 0:
				print(string)
		elif verbose == 2:
			print(string)
	#---------- Accuracy Functions ---------- #
	def Binary_Accuracy(self, y_true, y_pred):
		predictions = (y_pred > 0.5) * 1
		accuracy = np.mean(predictions==y_true)
		return(accuracy)
	#---------- Optimisers ---------- #
	def SGD(self, lr, layer):
		layer.w -= lr * layer.dL_dw
		layer.b -= lr * layer.dL_db
	#---------- Loss Functions ---------- #
	class BCE_Loss():
		''' The Binary Cross-Entropy loss function '''
		def forward(self, y_true, y_pred):
			''' The forward pass '''
			y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
			loss = -(y_true*np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
			return(loss)
		def backward(self, y_true,  y_pred):
			''' The backward pass '''
			y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
			DL_DY = (y_pred-y_true) / (y_pred * (1-y_pred)) / len(y_pred)
			return(DL_DY)
	#---------- Activation Functions ----------#
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
	#---------- Layers ----------#
	class Dense():
		''' A dense layer '''
		def __init__(self, inputs=1, outputs=1, alg='he uniform', sd=0.1, a=-0.5, b=0.5):
			''' Initialise parameters '''
			self.w, self.b, self.gamma, self.beta = self.Parameters(inputs, outputs, alg=alg, sd=sd, a=a, b=b)
			self.dL_dw, self.dL_db, self.dL_dx = None, None, None
		def Parameters(self, inputs=1, outputs=1, alg='he uniform', sd=0.1, a=-0.5, b=0.5):
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
		def forward(self, x):
			''' The forward pass '''
			self.x = x
			z = np.dot(self.x, self.w) + self.b
			return(z)
		def backward(self, DA_Dz):
			''' The backward pass '''
			self.dL_dw = np.dot(self.x.T, DA_Dz)
			self.dL_db = np.sum(DA_Dz, axis=0, keepdims=True)
			self.dL_dx = np.dot(DA_Dz, self.w.T)
			return(self.dL_dx)




	#---------- Training ---------- #
	def train(self, X, Y, loss, optimiser, lr, accuracy, batch_size=None, epochs=1, verbose=0):
		''' Train the network, forward pass followed by backward pass '''
		y_true = Y
		if loss == 'BCE':
			loss_fn = self.BCE_Loss()
		for epoch in range(epochs):
		# divide into Mini-Baches
		# steps in mini batches



			# Forward propagation
			output = X
			for layer in self.layers:
				output = layer.forward(output)
			y_pred = output
			cost = self.cost(loss_fn.forward(y_true, y_pred))
			# Tracking metric
			if accuracy == 'binary':
				A = self.Binary_Accuracy(y_true, y_pred)
			# Backpropagation
			grad = loss_fn.backward(y_true, y_pred)
			grad = self.layers[-1].backward(grad)
			for i in range(len(self.layers) - 2, -1, -1):
				grad = self.layers[i].backward(grad)
			# Gradient descent
			for layer in self.layers:
				if isinstance(layer, self.Dense):
					if optimiser == 'SGD':
						self.SGD(lr, layer)
			self.verbosity(epoch, cost, A, verbose=verbose)




























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
model.add(model.Dense(2, 64, alg='random normal'))
model.add(model.ReLU())
model.add(model.Dense(64, 1, alg='random normal'))
model.add(model.Sigmoid())





model.train(X, Y, 'BCE', 'SGD', 0.01, 'binary', epochs=200000, verbose=1)














