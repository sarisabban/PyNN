import math
#import pickle as pkl
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



#	def show(self):
#		''' Print out the structure of the network '''
#	def save(self, path):
#		''' Save model '''
#		#### serialise the self.layers object
#	def load(self, path):
#		''' Load model '''
#		#### Load the layers object into self.layers
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
	class Regression_Accuracy():
		''' Accuracy for regression models '''
		def calc(self, y_true, y_pred):
			accuracy_precision = np.std(y_true) / 250
			predictions = y_pred
			accuracy = np.mean(np.absolute(predictions-y_true) < accuracy_precision)
			return(accuracy)
	class Binary_Accuracy():
		''' Accuracy for binary classification models '''
		def calc(self, y_true, y_pred):
			predictions = (y_pred > 0.5) * 1
			accuracy = np.mean(predictions == y_true)
			return(accuracy)
	class Categorical_Accuracy():
		''' Accuracy for categorical classification models '''
		def calc(self, y_true, y_pred):
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
			if len(y_true.shape) == 1:
				y_pred_vector = y_pred[range(len(y_pred)), y_true]
			elif len(y_true.shape) == 2:
				y_pred_vector = np.sum(y_pred_clipped * y_true, axis=1)
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
	def SGD(self, lr, decay, iters, layer):
		''' The Stochastic Gradient Descent optimiser '''
		lr = lr * (1. / (1. + decay * iters))
		layer.w -= lr * layer.dL_dw
		layer.b -= lr * layer.dL_db
	def Adagrad(self, lr, decay, iters, e, layer):
		''' The Adagrad optimiser '''
		lr = lr * (1. / (1. + decay * iters))
		self.cache = {}
		if layer not in self.cache:
			w0 = np.zeros_like(layer.w)
			b0 = np.zeros_like(layer.b)
			self.cache[layer] = {'w':w0, 'b':b0}
		self.cache[layer]['w'] += layer.dL_dw ** 2
		self.cache[layer]['b'] += layer.dL_db ** 2
		layer.w -= (lr / (np.sqrt(self.cache[layer]['w']) + e)) * layer.dL_dw
		layer.b -= (lr / (np.sqrt(self.cache[layer]['b']) + e)) * layer.dL_db
	def RMSprop(self, lr, decay, iters, beta, e, layer):
		''' The RMSprop optimiser '''
		# Initialize cache if not already
		lr = lr * (1. / (1. + decay * iters))
		self.cache = {}
		if layer not in self.cache:
			w0 = np.zeros_like(layer.w)
			b0 = np.zeros_like(layer.b)
			self.cache[layer] = {'w':w0, 'b':b0}
		w_cache = beta * self.cache[layer]['w'] + (1 - beta) * layer.dL_dw ** 2
		b_cache = beta * self.cache[layer]['b'] + (1 - beta) * layer.dL_db ** 2
		self.cache[layer]['w'] = w_cache
		self.cache[layer]['b'] = b_cache
		layer.w -= (lr / (np.sqrt(self.cache[layer]['w']) + e)) * layer.dL_dw
		layer.b -= (lr / (np.sqrt(self.cache[layer]['b']) + e)) * layer.dL_db
	def Adam(self, lr, decay, iters, beta1, beta2, e, layer):
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
			''' Parameter initialisation function '''
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
	def train(self,
			X_train=None, Y_train=None,
			X_valid=None, Y_valid=None,
			X_tests=None, Y_tests=None,
			loss='BCE',
			optimiser='SGD', lr=0.1, decay=5e-7, beta1=0.9, beta2=0.999, e=1e-7,
			accuracy='BINARY',
			batch_size=None,
			epochs=1,
			verbose=1):
		''' Train the network, forward pass followed by backward pass '''
		steps = 0
		X_train_batch = X_train
		Y_train_batch = Y_train
		if   loss.lower() == 'bce': loss_fn = self.BCE_Loss()
		elif loss.lower() == 'cce': loss_fn = self.CCE_Loss()
		elif loss.lower() == 'mse': loss_fn = self.MSE_Loss()
		elif loss.lower() == 'mae': loss_fn = self.MAE_Loss()
		if   accuracy.lower() == 'regression': acc = self.Regression_Accuracy()
		elif accuracy.lower() == 'binary':     acc = self.Binary_Accuracy()
		elif accuracy.lower() == 'categorical':acc = self.Categorical_Accuracy()
		if batch_size is not None: steps = X_train.shape[0] // batch_size
		for epoch in range(epochs):
			############# EARLY STOPPING HERE #################
			for step in range(steps + 1):
				if batch_size is not None:
					X_train_batch = X[step*batch_size:(step+1)*batch_size]
					Y_train_batch = Y[step*batch_size:(step+1)*batch_size]
				# Forward propagation
				output = X_train_batch
				y_true = Y_train_batch
				for layer in self.layers: output = layer.forward(output)
				y_pred = output
				cost_train = self.cost(loss_fn.forward(y_true, y_pred))
				# Accuracy calculation
				accuracy_train = acc.calc(y_true, y_pred)
				# Backpropagation
				grad = loss_fn.backward(y_true, y_pred)
				grad = self.layers[-1].backward(grad)
				for i in range(len(self.layers) - 2, -1, -1):
					grad = self.layers[i].backward(grad)
				# Gradient descent
				for layer in self.layers:
					if isinstance(layer, self.Dense):
						if optimiser.lower() == 'sgd':
							self.SGD(lr, decay, epoch, layer)
						elif optimiser.lower() == 'adagrad':
							self.Adagrad(lr, decay, epoch, e, layer)
						elif optimiser.lower() == 'rmsprop':
							self.RMSprop(lr, decay, epoch, beta1, e, layer)
						elif optimiser.lower() == 'adam':
							self.Adam(lr, decay, epoch, beta1, beta2, e, layer)
				if verbose == 2:
					self.verbosity('train', epoch, step, cost_train, accuracy_train)
			if verbose == 1:
				self.verbosity('train', epoch, step, cost_train, accuracy_train)
			# Evaluate validation set
			if X_valid is not None and Y_valid is not None:
				output = X_valid
				y_true = Y_valid
				for layer in self.layers: output = layer.forward(output)
				y_pred = output
				cost_valid = self.cost(loss_fn.forward(y_true, y_pred))
				accuracy_valid = acc.calc(y_true, y_pred)
				if verbose == 1 or verbose == 2:
					self.verbosity('validation', epoch, step, cost_valid, accuracy_valid)
		# Evaluate test set
		if X_tests is not None and Y_tests is not None:
			output = X_tests
			y_true = Y_tests
			for layer in self.layers: output = layer.forward(output)
			y_pred = output
			cost_tests = self.cost(loss_fn.forward(y_true, y_pred))
			accuracy_tests = acc.calc(y_true, y_pred)
			if verbose == 1 or verbose == 2:
				self.verbosity('test', epoch, step, cost_tests, accuracy_tests)




# Add early stopping
# Regularisation


	def verbosity(self, sets, E, S, C, A):
		''' Control level of information printout during training '''
		if sets.lower() == 'train':
			s1 = f'Set: Training |'
			s2 = f'Epoch: {E:,} | Batch: {S:,} |'
			s3 = f'Cost: {C:.5f} | Accuracy: {A:.5f}'
			string = s1 + s2 + s3
		elif sets.lower() == 'validation':
			s1 = f'Set: Validation | '
			s2 = f'Epoch: {E:,} | '
			s3 = f'Cost: {C:.5f} | Accuracy: {A:.5f}'
			string = s1 + s2 + s3
		elif sets.lower() == 'test':
			string = f'Set: Test | Cost: {C:.5f} | Accuracy: {A:.5f}'
		print(string)









import sklearn


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

X, Y = spiral_data(samples=140, classes=2)
Y = Y.reshape(-1, 1)

X_train, X_valid, Y_train, Y_valid = sklearn.model_selection.train_test_split(X, Y, train_size=200)
X_valid, X_tests, Y_valid, Y_tests = sklearn.model_selection.train_test_split(X_valid, Y_valid, train_size=40)



model = PyNN()
model.add(model.Dense(2, 64))
model.add(model.ReLU())
model.add(model.Dense(64, 1))
model.add(model.Sigmoid())

model.train(X_train, Y_train, X_valid, Y_valid, X_tests, Y_tests, batch_size=16, epochs=200, verbose=2)
