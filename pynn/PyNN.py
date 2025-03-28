import time
import math
import pickle
#import cupy as cp
import numpy as np

np.random.seed(42)

class PyNN():
	''' Lightweight NumPy-based neural network library '''
	#---------- Utilities ----------#
	def __init__(self):
		''' Initialise the class with the following objects and detect GPU '''
		self.layers = []
		self.R = '\033[31m' # Red
		self.G = '\033[32m' # Green
		self.B = '\033[34m' # Blue
		self.O = '\033[33m' # Orange
		self.g = '\033[90m' # Grey
		self.W = '\033[97m' # White
		self.P = '\033[35m' # Purple
		self.r = '\033[0m'  # Reset
		try:
			cp.cuda.Device(0).compute_capability
			self.chip = 'GPU'
		except:
			self.chip = 'CPU'
		print(f'\x1B[3m\033[1m{self.P}PyNN{self.r} ', end='')
		print(f'is running on {self.g}{self.chip}{self.r}')
	def add(self, layer):
		''' Add a layer to the network '''
		if isinstance(layer, self.Dense):
			layer.chip = self.chip
		self.layers.append(layer)
	def shuffle_data(self, X, Y):
		''' Shuffle X and Y in unison '''
		indices = np.random.permutation(X.shape[0])
		return(X[indices], Y[indices])
	def cost(self, loss):
		''' The cost function '''
		return(np.mean(loss))
	def forward(self, X):
		''' Forward propagation '''
		for layer in self.layers:
			X = layer.forward(X)
		output = X
		return(output)
	def backward(self, dy):
		''' Backpropagation '''
		dx = self.layers[-1].backward(dy)
		for i in range(len(self.layers) - 2, -1, -1):
			dx = self.layers[i].backward(dx)
		return(dx)
	def predict(self, X):
		''' Perform a prediction '''
		for layer in self.layers:
			if isinstance(layer, (self.Dropout)): pass
			else: X = layer.forward(X)
		y_pred = X
		return(y_pred)
	class EarlyStopping():
		''' Early stopping function tracking training loss stagnation '''
		def __init__(self, min_delta=1e-4):
			self.min_delta = min_delta
			self.best_loss = float('inf')
		def check(self, epoch, loss):
			if loss < self.best_loss - self.min_delta: self.best_loss = loss
			if epoch > 1 and abs(loss - self.best_loss) < self.min_delta:
				print('{self.R}Early Stop: Training loss plateaued{self.r}')
				return(True)
			return(False)
	def show(self):
		''' Print out the structure of the network '''
		print(f'{self.B}' + '-'*60 + f'{self.r}')
		print(f"{self.g}{'Layer':<25}{'Shape':<25}Parameters{self.r}")
		print(f'{self.B}' + '-'*60 + f'{self.r}')
		total_params = 0
		for layer in self.layers:
			name = layer.__class__.__name__
			if hasattr(layer, 'w'):
				shape = layer.w.shape
				params = math.prod(shape)
				total_params += params
			else:
				shape, params = '', ''
			print(f'{self.O}{name:<25}{self.G}{str(shape):<25}{params}{self.r}')
		print(f'{self.B}' + '-'*30 + f'{self.r}')
		print(f'{self.P}Total Parameters: \033[1m{total_params:,}{self.r}')
	def verbosity(self, sets, cost, accuracy, args=[]):
		''' Print training information '''
		C = f'{self.O}{cost:.5f}{self.r}'
		A = f'{self.O}{accuracy:.5f}{self.r}'
		h = f'{self.P}-{self.r}'
		t = f'{self.g}{args[4]:.0f}s{self.r}'
		B = f'  Batch {self.B}{args[2]}/{args[3]}{self.r}'
		V = f'{self.g}{sets} epoch {args[0]}/{args[1]}:{self.r}'
		R = f'{self.g}{sets}:{self.r}'
		if sets.lower() == 'train':
			string = f'{B} {h} Cost {C} {h} Accuracy {A} {h} {t}'
		elif sets.lower() == 'valid':
			string = f'{V} {h} Cost {C} {h} Accuracy {A} {h} {t}'
		elif sets.lower() == 'tests':
			string = f'{R} Cost {C} {h} Accuracy {A} {h} {t}'
		print(string)
	def save(self, path='./model'):
		''' Save model '''
		with open(f'{path}.pkl', 'wb') as f:
			pickle.dump(self.layers, f)
	def load(self, path='./model'):
		''' Load model '''
		with open(f'{path}.pkl', 'rb') as f:
			self.layers = pickle.load(f)
	def flatten(self, X):
		''' Flattens a layer to 1D '''
		X = X.flatten()
		return(X)
	#---------- Activation Functions ----------#
	class Step():
		''' The Step activation function (for binary classification) '''
		def forward(self, z):
			y = np.where(z >= 0, 1, 0)
			return(y)
		def backward(self, dy):
			return(np.zeros_like(dy))
	class Linear():
		''' The Linear activation function '''
		def forward(self, z):
			y = z
			return(y)
		def backward(self, dy):
			dz = dy.copy()
			return(dz)
	class Sigmoid():
		''' The Sigmoid activation function '''
		def forward(self, z):
			self.y = 1 / (1 + np.exp(-z))
			return(self.y)
		def backward(self, dy):
			dz = dy * (1 - self.y) * self.y
			return(dz)
	class ReLU():
		''' The ReLU activation function '''
		def forward(self, z):
			self.z = z
			y = np.maximum(0, self.z)
			return(y)
		def backward(self, dy):
			dz = dy.copy()
			dz[self.z <= 0] = 0
			return(dz)
	class LeakyReLU():
		''' The LeakyReLU activation function '''
		def __init__(self, alpha=0.01):
			self.alpha = alpha
		def forward(self, z):
			self.z = z
			y = np.where(z > 0, z, self.alpha * z)
			return(y)
		def backward(self, dy):
			dz = dy.copy()
			dz[self.z <= 0] *= self.alpha
			return(dz)
	class TanH():
		''' The Hyperbolic Tangent (TanH) activation function '''
		def forward(self, z):
			self.y = np.tanh(z)
			return(self.y)
		def backward(self, dy):
			dz = dy * (1 - self.y ** 2)
			return(dz)
	class Softmax():
		''' The Softmax activation function '''
		def forward(self, z):
			exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
			self.y = exp_values / np.sum(exp_values, axis=1, keepdims=True)
			return(self.y)
		def backward(self, dY):
			dz = np.empty_like(dY)
			for i, (y, dy) in enumerate(zip(self.y, dY)):
				y = y.reshape(-1, 1)
				jacobian_matrix = np.diagflat(y) - np.dot(y, y.T)
				dz[i] = np.dot(jacobian_matrix, dy)
			return(dz)
	#---------- Loss Functions ----------#
	class MSE_Loss():
		''' The Mean Squared Error loss '''
		def forward(self, y_true, y_pred):
			loss = (y_true - y_pred)**2
			return(loss)
		def backward(self, y_true, y_pred):
			dy = (-2 * (y_true - y_pred) / len(y_pred[0])) / len(y_pred)
			return(dy)
	class MAE_Loss():
		''' The Mean Absolute Error loss '''
		def forward(self, y_true, y_pred):
			loss = np.abs(y_true - y_pred)
			return(loss)
		def backward(self, y_true, y_pred):
			dy = (np.sign(y_true - y_pred) / len(y_pred[0])) / len(y_pred)
			return(dy)
	class BCE_Loss():
		''' The Binary Cross-Entropy loss function '''
		def forward(self, y_true, y_pred):
			y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
			loss = -(y_true*np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
			return(loss)
		def backward(self, y_true,  y_pred):
			y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
			dy=-(y_true/y_pred-(1-y_true)/(1-y_pred))/len(y_pred[0])/len(y_pred)
			return(dy)
	class CCE_Loss():
		''' The Categorical Cross-Entropy loss '''
		def forward(self, y_true, y_pred):
			y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
			if len(y_true.shape) == 1:
				y_pred_vector = y_pred[range(len(y_pred)), y_true]
			elif len(y_true.shape) == 2:
				y_pred_vector = np.sum(y_pred * y_true, axis=1)
			loss = -np.log(y_pred_vector)
			return(loss)
		def backward(self, y_true,  y_pred):
			if len(y_true.shape) == 1:
				y_true = np.eye(len(y_pred[0]))[y_true]
			dy = (-y_true / y_pred) / len(y_pred)
			return(dy)
	#---------- Accuracy Functions ----------#
	class Regression_Accuracy():
		''' Accuracy for regression models '''
		def calc(self, y_true, y_pred):
			precision = np.std(y_true) / 250
			predictions = y_pred
			accuracy = np.mean(np.absolute(predictions-y_true) < precision)
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
	#---------- Optimisers ----------#
	def SGD(self, lr, decay, iters, layer):
		''' The Stochastic Gradient Descent optimiser '''
		lr = lr * (1. / (1. + decay * iters))
		layer.w += -lr * layer.dw
		layer.b += -lr * layer.db
	def Adagrad(self, lr, decay, iters, e, layer):
		''' The Adagrad optimiser '''
		lr = lr * (1. / (1. + decay * iters))
		self.cache = {}
		if layer not in self.cache:
			w0 = np.zeros_like(layer.w)
			b0 = np.zeros_like(layer.b)
			self.cache[layer] = {'w':w0, 'b':b0}
		self.cache[layer]['w'] += layer.dw ** 2
		self.cache[layer]['b'] += layer.db ** 2
		layer.w -= (lr / (np.sqrt(self.cache[layer]['w']) + e)) * layer.dw
		layer.b -= (lr / (np.sqrt(self.cache[layer]['b']) + e)) * layer.db
	def RMSprop(self, lr, decay, iters, beta, e, layer):
		''' The RMSprop optimiser '''
		lr = lr * (1. / (1. + decay * iters))
		self.cache = {}
		if layer not in self.cache:
			w0 = np.zeros_like(layer.w)
			b0 = np.zeros_like(layer.b)
			self.cache[layer] = {'w':w0, 'b':b0}
		w_cache = beta * self.cache[layer]['w'] + (1 - beta) * layer.dw ** 2
		b_cache = beta * self.cache[layer]['b'] + (1 - beta) * layer.db ** 2
		self.cache[layer]['w'] = w_cache
		self.cache[layer]['b'] = b_cache
		layer.w -= (lr / (np.sqrt(self.cache[layer]['w']) + e)) * layer.dw
		layer.b -= (lr / (np.sqrt(self.cache[layer]['b']) + e)) * layer.db
	def Adam(self, lr, decay, iters, beta1, beta2, e, layer):
		''' The Adam Gradient Descent optimiser '''
		lr = lr * (1. / (1. + decay * iters))
		layer.w_m = beta1 * layer.w_m + (1 - beta1) * layer.dw
		layer.b_m = beta1 * layer.b_m + (1 - beta1) * layer.db
		w_m_c = layer.w_m / (1 - beta1 ** (iters + 1))
		b_m_c = layer.b_m / (1 - beta1 ** (iters + 1))
		layer.w_c = beta2 * layer.w_c + (1 - beta2) * layer.dw**2
		layer.b_c = beta2 * layer.b_c + (1 - beta2) * layer.db**2
		w_c_c = layer.w_c / (1 - beta2 ** (iters + 1))
		b_c_c = layer.b_c / (1 - beta2 ** (iters + 1))
		layer.w -= lr * w_m_c / (np.sqrt(w_c_c) + e)
		layer.b -= lr * b_m_c / (np.sqrt(b_c_c) + e)
	#---------- Regularisation ----------#
	class Dropout():
		''' The dropout regularisation layer '''
		def __init__(self, p=0.25):
			self.p = p
		def forward(self, y, train=True):
			if train:
				self.mask = np.random.binomial(1, 1-self.p, y.shape)/(1-self.p)
				y *= self.mask
			return(y)
		def backward(self, dz):
			dz *= self.mask
			return(dz)
	class BatchNorm():
		''' The Batch Normalisation regularisation layer '''
		def __init__(self, g=1.0, b=0.0, e=1e-7):
			self.w = g
			self.b = b
			self.e = e
		def forward(self, z):
			self.z = z
			self.mean = np.mean(z, axis=0, keepdims=True)
			self.var = np.var(z, axis=0, keepdims=True)
			self.z_norm = (self.z - self.mean) / np.sqrt(self.var + self.e)
			z_new = self.w * self.z_norm + self.b
			return(z_new)
		def backward(self, dy):
			m = self.z.shape[0]
			self.dw = np.sum(dy * self.z_norm, axis=0, keepdims=True) # dg
			self.db = np.sum(dy, axis=0, keepdims=True)               # db
			self.dz = (self.w * (1./np.sqrt(self.var + self.e)) / m) \
			* (m*dy - np.sum(dy, axis=0) - (1./np.sqrt(self.var + self.e))**2 \
			* (self.z - self.mean) * np.sum(dy*(self.z - self.mean), axis=0))
			return(self.dz)
	#---------- Layers ----------#
	class Dense():
		''' A dense layer '''
		def __init__(self, inputs=1, outputs=1,
					alg='he uniform', mean=0.0, sd=0.1, a=-0.5, b=0.5,
					l1w=0, l1b=0, l2w=0, l2b=0):
			''' Initialise parameters '''
			self.l1w, self.l1b, self.l2w, self.l2b = l1w, l1b, l2w, l2b
			self.Parameters(inputs, outputs, alg, mean, sd, a, b)
		def Parameters(self, inputs, outputs, alg, mean, sd, a, b):
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
			if self.chip == 'CPU':
				self.L1w = self.l1w * np.sum(np.abs(self.w))
				self.L1b = self.l1b * np.sum(np.abs(self.b))
				self.L2w = self.l2w * np.sum(self.w**2)
				self.L2b = self.l2b * np.sum(self.b**2)
				z = np.dot(self.x, self.w) + self.b
			elif self.chip == 'GPU':
				self.L1w = self.l1w * cp.sum(np.abs(self.w))
				self.L1b = self.l1b * cp.sum(np.abs(self.b))
				self.L2w = self.l2w * cp.sum(self.w**2)
				self.L2b = self.l2b * cp.sum(self.b**2)
				z = cp.dot(self.x, self.w) + self.b
			self.L1L2 = self.L1w + self.L1b + self.L2w + self.L2b
			return(z)
		def backward(self, dz):
			if self.chip == 'CPU':
				self.dw = np.dot(self.x.T, dz)
				self.db = np.sum(dz, axis=0, keepdims=True)
				self.dx = np.dot(dz, self.w.T)
				L1_dw = np.ones_like(self.w)
				L1_dw[self.w < 0] = -1
				self.dw += self.l1w * L1_dw
				L1_db = np.ones_like(self.b)
				L1_db[self.b < 0] = -1
				self.db += self.l1b * L1_db
				self.dw += 2 * self.l2w * self.w
				self.db += 2 * self.l2b * self.b
			elif self.chip == 'GPU':
				self.dw = cp.dot(self.x.T, dz)
				self.db = cp.sum(dz, axis=0, keepdims=True)
				self.dx = cp.dot(dz, self.w.T)
				L1_dw = cp.ones_like(self.w)
				L1_dw[self.w < 0] = -1
				self.dw += self.l1w * L1_dw
				L1_db = cp.ones_like(self.b)
				L1_db[self.b < 0] = -1
				self.db += self.l1b * L1_db
				self.dw += 2 * self.l2w * self.w
				self.db += 2 * self.l2b * self.b
			return(self.dx)
	#---------- Training ----------#
	def train(self,
			X_train=None, Y_train=None,
			X_valid=None, Y_valid=None,
			X_tests=None, Y_tests=None,
			batch_size=None,
			loss='BCE',
			accuracy='BINARY',
			optimiser='SGD', lr=0.1, decay=0.0, beta1=0.9, beta2=0.999, e=1e-7,
			early_stop=False,
			epochs=1,
			verbose=1):
		''' Train the network '''
		steps = 0
		if   loss.lower() == 'mse': loss_fn = self.MSE_Loss()
		elif loss.lower() == 'mae': loss_fn = self.MAE_Loss()
		elif loss.lower() == 'bce': loss_fn = self.BCE_Loss()
		elif loss.lower() == 'cce': loss_fn = self.CCE_Loss()
		if   accuracy.lower() == 'regression': acc = self.Regression_Accuracy()
		elif accuracy.lower() == 'binary':     acc = self.Binary_Accuracy()
		elif accuracy.lower() == 'categorical':acc = self.Categorical_Accuracy()
		if batch_size is not None: steps = X_train.shape[0] // batch_size
		if early_stop: STOP = self.EarlyStopping()
		for epoch in range(epochs):
			if verbose == 2:
				T = f'{self.g}Training epoch {epoch+1}/{epochs}:{self.r}'
				print(f'{T}')
			# Shuffle training datatset
			X_train, Y_train = self.shuffle_data(X_train, Y_train)
			for step in range(steps + 1):
				start = time.process_time()
				# Batch segmentation
				X_train_batch = X_train
				Y_train_batch = Y_train
				if batch_size is not None:
					X_train_batch = X[step * batch_size:(step + 1) * batch_size]
					Y_train_batch = Y[step * batch_size:(step + 1) * batch_size]
				# Forward propagation
				y_true = Y_train_batch
				y_pred = self.forward(X_train_batch)
				L1L2 = [l.L1L2 for l in self.layers if isinstance(l,self.Dense)]
				L1L2 = sum(L1L2)
				cost_train = self.cost(loss_fn.forward(y_true, y_pred)) + L1L2
				accuracy_train = acc.calc(y_true, y_pred)
				# Backpropagation
				dy = loss_fn.backward(y_true, y_pred)
				dx = self.backward(dy)
				# Gradient descent
				for layer in self.layers:
					if isinstance(layer, (self.Dense)):
						if optimiser.lower() == 'sgd':
							self.SGD(lr, decay, epoch, layer)
						elif optimiser.lower() == 'adagrad':
							self.Adagrad(lr, decay, epoch, e, layer)
						elif optimiser.lower() == 'rmsprop':
							self.RMSprop(lr, decay, epoch, beta1, e, layer)
						elif optimiser.lower() == 'adam':
							self.Adam(lr, decay, epoch, beta1, beta2, e, layer)
					end = time.process_time()
					t = end - start
				if verbose == 2:
					args = [epoch + 1, epochs, step + 1, steps + 1, t]
					self.verbosity('Train', cost_train, accuracy_train, args)
			if early_stop and STOP.check(epoch, cost_train): break
			# Evaluate validation set
			if X_valid is not None and Y_valid is not None:
				start = time.process_time()
				y_true = Y_valid
				y_pred = self.predict(X_valid)
				cost_valid = self.cost(loss_fn.forward(y_true, y_pred))
				accuracy_valid = acc.calc(y_true, y_pred)
				end = time.process_time()
				t = end - start
				if verbose == 1 or verbose == 2:
					args = [epoch + 1, epochs, step + 1, steps + 1, t]
					self.verbosity('Valid', cost_valid, accuracy_valid, args)
		# Evaluate test set
		if X_tests is not None and Y_tests is not None:
			start = time.process_time()
			y_true = Y_tests
			y_pred = self.predict(X_tests)
			cost_tests = self.cost(loss_fn.forward(y_true, y_pred))
			accuracy_tests = acc.calc(y_true, y_pred)
			end = time.process_time()
			t = end - start
			if verbose == 1 or verbose == 2:
				self.verbosity('Tests', cost_tests, accuracy_tests, args)





#----- Import Data -----#
import sklearn
def sine_data(samples=1000):
	X = np.arange(samples).reshape(-1, 1) / samples
	y = np.sin(2 * np.pi * X).reshape(-1, 1)
	return(X, y)

X, Y = sine_data()

X_train, X_valid, Y_train, Y_valid = sklearn.model_selection.train_test_split(X, Y, train_size=600)
X_valid, X_tests, Y_valid, Y_tests = sklearn.model_selection.train_test_split(X, Y, train_size=200)

model = PyNN()
model.add(model.Dense(1, 64))
model.add(model.Sigmoid())
model.add(model.Dense(64, 64))
model.add(model.Sigmoid())
model.add(model.Dense(64, 1))
model.add(model.Linear())

model.show()

model.train(
X_train, Y_train,
X_valid, Y_valid,
X_tests, Y_tests,
loss='MSE', accuracy='regression', batch_size=32, lr=0.05, epochs=1, verbose=2)
