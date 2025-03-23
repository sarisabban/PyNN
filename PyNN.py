import math
import numpy as np
np.random.seed(42)

class PyNN():
	''' Lightweight NumPy-based neural network library '''
	#---------- Utilities ---------- #
	def __init__(self):
		''' Initialise the class with the following objects '''
		self.layers = []
	def add(self, layer):
		''' Add a layer to the network '''
		self.layers.append(layer)
	def cost(self, loss):
		''' The cost function '''
		return(np.mean(loss))
	def verbosity(self, epoch, cost, accuracy, *, verbose=1):
		''' Control level of information printout during training '''
		E, C, A = epoch, cost, accuracy
		string = f'Epoch: {E:,} | Cost: {C:.5f} | Accuracy: {A:.5f}'
		if verbose == 0:
			pass
		elif verbose == 1:
			if epoch % 100 == 0: print(string)
		elif verbose == 2:
			print(string)


#	def show(self):
#		''' Print out the structure of the network '''
#	def load(self, path):
#		''' Load model '''
#	def save(self, path):
#		''' Save model '''
#		with open(path, 'wb') as f:
#		pickle.dump(self.w, f)
#	def predict(self):
#	''' Perform a prediction '''
#	def GPU(self):
#	''' Use GPU instead of CPU '''






	#---------- Activation Functions ----------#
	class Step():
		''' The Step activation function (for binary classification) '''
		def forward(self, z):
			self.y = np.where(z >= 0, 1, 0)
			return self.y
		def backward(self, DL_DY):
			return(np.zeros_like(DL_DY))
	class Linear():
		''' The Linear activation function '''
		def forward(self, z):
			self.y = z
			return(self.y)
		def backward(self, DL_DY):
			DA_Dz = DL_DY.copy()
			return(DA_Dz)
	class Sigmoid():
		''' The Sigmoid activation function '''
		def forward(self, z):
			self.y = 1 / (1 + np.exp(-z))
			return(self.y)
		def backward(self, DA_Dz):
			DS_Dz = DA_Dz * (1 - self.y) * self.y
			return(DS_Dz)
	class ReLU():
		''' The ReLU activation function '''
		def forward(self, z):
			self.z = z
			y = np.maximum(0, self.z)
			return(y)
		def backward(self, DL_DY):
			DA_Dz = DL_DY.copy()
			DA_Dz[self.z <= 0] = 0
			return(DA_Dz)
	class LeakyReLU():
		''' The LeakyReLU activation function '''
		def __init__(self, alpha=0.01):
			self.alpha = alpha
		def forward(self, z):
			self.z = z
			y = np.where(z > 0, z, self.alpha * z)
			return(y)
		def backward(self, DL_DY):
			DA_Dz = DL_DY.copy()
			DA_Dz[self.z <= 0] *= self.alpha
			return(DA_Dz)
	class TanH():
		''' The Hyperbolic Tangent (TanH) activation function '''
		def forward(self, z):
			self.y = np.tanh(z)
			return(self.y)
		def backward(self, DL_DY):
			DT_Dz = (1 - self.y ** 2)
			return(DL_DY * DT_Dz)
	class Softmax():
		''' The Softmax activation function '''
		def forward(self, z):
			exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
			self.y = exp_values / np.sum(exp_values, axis=1, keepdims=True)
			return(self.y)
		def backward(self, DL_DY):
			DS_Dz = np.empty_like(DL_DY)
			for i, (y, dy) in enumerate(zip(self.y, DL_DY)):
				y = y.reshape(-1, 1)
				jacobian_matrix = np.diagflat(y) - np.dot(y, y.T)
				DS_Dz[i] = np.dot(jacobian_matrix, dy)
			return(DS_Dz)
	#---------- Accuracy Functions ---------- #
	def Regression_Accuracy(self, y_true, y_pred):
		''' Accuracy for regression models '''
		accuracy_precision = np.std(y_true) / 250
		predictions = y_pred
		accuracy = np.mean(np.absolute(predictions-y_true) < accuracy_precision)
		return(accuracy)
	def Binary_Accuracy(self, y_true, y_pred):
		''' Accuracy for binary classification models '''
		predictions = (y_pred > 0.5) * 1
		accuracy = np.mean(predictions == y_true)
		return(accuracy)
	def Categorical_Accuracy(self, y_true, y_pred):
		''' Accuracy for categorical classification models '''
		predictions = np.argmax(y_pred, axis=1)
		accuracy = np.mean(predictions == y_true)
		return(accuracy)
	#---------- Loss Functions ---------- #
	class BCE_Loss():
		''' The Binary Cross-Entropy loss function '''
		def forward(self, y_true, y_pred):
			y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
			loss = -(y_true*np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
			return(loss)
		def backward(self, y_true,  y_pred):
			y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
			DL_DY = (y_pred - y_true) / (y_pred * (1 - y_pred)) / len(y_pred)
			return(DL_DY)
	class CCE_Loss():
		''' The Categorical Cross-Entropy loss '''
		def forward(self, y_true, y_pred):
			y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
			y_pred_vector = y_pred[range(len(y_pred)), y_true]
			loss = -np.log(y_pred_vector)
			return(loss)
		def backward(self, y_true,  y_pred):
			if len(y_true.shape) == 1:
				y_true = np.eye(len(y_pred[0]))[y_true]
			DL_DY = (-y_true / y_pred) / len(y_pred)
			return(DL_DY)
	class MSE_Loss():
		''' The Mean Squared Error loss '''
		def forward(self, y_true, y_pred):
			loss = (y_true - y_pred)**2
			return(loss)
		def backward(self, y_true, y_pred):
			DL_DY = (-2 * (y_true - y_pred) / len(y_pred[0])) / len(y_pred)
			return(DL_DY)
	class MAE_Loss():
		''' The Mean Absolute Error loss '''
		def forward(self, y_true, y_pred):
			loss = np.abs(y_true - y_pred)
			return(loss)
		def backward(self, y_true, y_pred):
			DL_DY = (np.sign(y_true - y_pred) / len(y_pred[0])) / len(y_pred)
			return(DL_DY)
	#---------- Optimisers ---------- #
	def SGD(self, lr, layer):
		''' The Stochastic Gradient Descent optimiser '''
		layer.w -= lr * layer.dL_dw
		layer.b -= lr * layer.dL_db
	def Adam(self, lr, decay, beta1, beta2, e, iters, layer):
		''' The Adam Gradient Descent optimiser '''
		lr = lr * (1. / (1. + decay * iters))
		layer.w_m = beta1 * layer.w_m + (1 - beta1) * layer.dL_dw
		layer.b_m = beta1 * layer.b_m + (1 - beta1) * layer.dL_db
		w_m_c = layer.w_m / (1 - beta1 ** (iters + 1))
		b_m_c = layer.b_m / (1 - beta1 ** (iters + 1))
		layer.w_c = beta2 * layer.w_c + (1 - beta2) * layer.dL_dw**2
		layer.b_c = beta2 * layer.b_c + (1 - beta2) * layer.dL_db**2
		w_c_c = layer.w_c / (1 - beta2 ** (iters + 1))
		b_c_c = layer.b_c / (1 - beta2 ** (iters + 1))
		layer.w -= lr * w_m_c / (np.sqrt(w_c_c) + e)
		layer.b -= lr * b_m_c / (np.sqrt(b_c_c) + e)
	#---------- Layers ----------#
	class Dense():
		''' A dense layer '''
		def __init__(self, inputs=1, outputs=1,
					alg='he uniform', mean=0.0, sd=0.1, a=-0.5, b=0.5):
			''' Initialise parameters '''
			self.Parameters(inputs, outputs, alg=alg, sd=sd, a=a, b=b)
			self.dL_dw, self.dL_db, self.dL_dx = None, None, None
		def Parameters(self, inputs=1, outputs=1,
					alg='he uniform', mean=0.0, sd=0.1, a=-0.5, b=0.5):
			''' Parameter Initialisation '''
			if alg == 'zeros':
				w = np.zeros((inputs, outputs))
			elif alg == 'ones':
				w = np.ones((inputs, outputs))
			elif alg == 'random normal':
				w = np.random.normal(loc=mean, scale=sd, size=(inputs, outputs))
			elif alg == 'random uniform':
				w = np.random.uniform(low=a, high=b, size=(inputs, outputs))
			elif alg == 'glorot normal':
				sd = 1 / (math.sqrt(inputs + outputs))
				w =  np.random.normal(loc=mean, scale=sd, size=(inputs,outputs))
			elif alg == 'glorot uniform':
				a = - math.sqrt(6) / (math.sqrt(inputs + outputs))
				b = math.sqrt(6) / (math.sqrt(inputs + outputs))
				w = np.random.uniform(low=a, high=b, size=(inputs, outputs))
			elif alg == 'he normal':
				sd = 2 / (math.sqrt(inputs))
				w = np.random.normal(loc=mean, scale=sd, size=(inputs, outputs))
			elif alg == 'he uniform':
				a = - math.sqrt(6) / (math.sqrt(inputs))
				b = math.sqrt(6) / (math.sqrt(inputs))
				w = np.random.uniform(low=a, high=b, size=(inputs, outputs))
			self.w = w
			self.b = np.zeros((1, outputs))
			self.beta = np.zeros((1, outputs))
			self.gamma = np.ones((1, outputs))
			self.w_m = np.zeros_like(w)
			self.w_c = np.zeros_like(w)
			self.b_m = np.zeros_like(b)
			self.b_c = np.zeros_like(b)
		def forward(self, x):
			self.x = x
			z = np.dot(self.x, self.w) + self.b
			return(z)
		def backward(self, DA_Dz):
			self.dL_dw = np.dot(self.x.T, DA_Dz)
			self.dL_db = np.sum(DA_Dz, axis=0, keepdims=True)
			self.dL_dx = np.dot(DA_Dz, self.w.T)
			return(self.dL_dx)









	#---------- Training ---------- #
	def train(self, X, Y, loss, optimiser, lr, accuracy, batch_size=None, epochs=1, verbose=0):
		''' Train the network, forward pass followed by backward pass '''
		y_true = Y
		if loss.upper() == 'BCE':
			loss_fn = self.BCE_Loss()
		for epoch in range(epochs):
		# divide into Mini-Baches
#		steps = X.shape[0] // batch_size
		# steps in mini batches



			# Forward propagation
			output = X
			for layer in self.layers:
				output = layer.forward(output)
			y_pred = output
			cost = self.cost(loss_fn.forward(y_true, y_pred))
			# Tracking metric
			if accuracy.upper() == 'BINARY':
				A = self.Binary_Accuracy(y_true, y_pred)
			# Backpropagation
			grad = loss_fn.backward(y_true, y_pred)
			grad = self.layers[-1].backward(grad)
			for i in range(len(self.layers) - 2, -1, -1):
				grad = self.layers[i].backward(grad)
			# Gradient descent
			for layer in self.layers:
				if isinstance(layer, self.Dense):
					if optimiser.upper() == 'SGD':
						self.SGD(lr, layer)
					elif optimiser.upper() == 'ADAM':

						decay, beta1, beta2, e, = 5e-7, 0.9, 0.999, 1.e-7

						self.Adam(lr, decay, beta1, beta2, e, epoch, layer)
			self.verbosity(epoch, cost, A, verbose=verbose)





#	''' Train on training set, then validate on valid, after training test of test set '''
#	''' Add early stopping '''























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

X, Y = spiral_data(samples=100, classes=2)
Y = Y.reshape(-1, 1)

model = PyNN()
model.add(model.Dense(2, 64))
model.add(model.ReLU())
model.add(model.Dense(64, 1))
model.add(model.Sigmoid())
model.train(X, Y, 'BCE', 'Adam', 0.01, 'binary', epochs=200000, verbose=1)
