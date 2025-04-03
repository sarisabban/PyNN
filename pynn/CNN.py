import time
import math
import pickle
import numpy as np

try: import cupy as cp
except: pass

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
				print(f'{self.R}Early Stop: Training loss plateaued{self.r}')
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
			if isinstance(layer, (self.Dense)):
				shape = layer.w.shape
				params = math.prod(shape)
				total_params += params
			else:
				shape, params = '', ''
			print(f'{self.O}{name:<25}{self.G}{str(shape):<25}{params}{self.r}')
		print(f'{self.B}' + '-'*30 + f'{self.r}')
		print(f'{self.P}Total Parameters: {total_params:,}{self.r}\n')
	def verbosity(self, sets, args):
		''' Print training information '''
		Cbatch, Abatch = args['cost_batch'], args['accuracy_batch']
		Ctrain, Atrain = args['cost_train'], args['accuracy_train']
		Cvalid, Avalid = args['cost_valid'], args['accuracy_valid']
		Ctests, Atests = args['cost_tests'], args['accuracy_tests']
		epoch,  epochs = args['epoch'],      args['epochs']
		step,   steps  = args['step'],       args['steps']
		time           = args['time']
		S = f'{self.g}{sets}:{self.r}'
		E = f'{self.g}epoch {epoch}/{epochs}{self.r}'
		h = f'{self.B}|{self.r}'
		n = f'{self.P}-{self.r}'
		t = f'{self.g}{time:.0f}s{self.r}'
		if sets.lower() == 'batch':
			Cb = f'{self.G}Cost {self.O}{Cbatch:.5f}{self.r}'
			Ab = f'{self.G}Accuracy {self.O}{Abatch:.5f}{self.r}'
			St = f'{self.g}batch {step}/{steps}{self.r}'
			string = f'{E} {n} {St} {n} {Cb} {n} {Ab} {n} {t}'
			print(string, end='\r')
		elif sets.lower() == 'train' or sets.lower() == 'valid':
			pass
			Ct = f'{self.G}Train Cost {self.O}{Ctrain:.5f}{self.r}'
			At = f'{self.G}Train Accuracy {self.O}{Atrain:.5f}{self.r}'
			if Cvalid != None:
				Cv = f'{self.G}Valid Cost {self.O}{Cvalid:.5f}{self.r}'
				Av = f'{self.G}Valid Accuracy {self.O}{Avalid:.5f}{self.r}'
				string = f'{S} {E:<30} {Ct} {h} {At} {h} {Cv} {h} {Av} {h} {t}'
			else:
				string = f'{S} {E:<30} {Ct} {h} {At} {h} {t}'
			print(string)
		elif sets.lower() == 'tests':
			Ct = f'{self.G}Tests Cost {self.O}{Ctests:.5f}{self.r}'
			At = f'{self.G}Tests Accuracy {self.O}{Atests:.5f}{self.r}'
			string = f'{S:<37} {Ct} {h} {At} {h} {t}'
			print(string)
	def save(self, path='./model.pkl'):
		''' Save model '''
		with open(f'{path}.pkl', 'wb') as f:
			pickle.dump(self.layers, f)
	def load(self, path='./model.pkl'):
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
			loss = np.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
			return(loss)
		def backward(self, y_true,  y_pred):
			y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
			dy=-(y_true/y_pred-(1-y_true)/(1-y_pred))/len(y_pred[0])/len(y_pred)
			return(dy)
	class CCE_Loss():
		''' The Categorical Cross-Entropy loss '''
		def forward(self, y_true, y_pred):
			y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
			y_true = y_true.astype(int)
			if len(y_true.shape) == 1:
				y_pred_vector = y_pred[range(len(y_pred)), y_true]
			elif len(y_true.shape) == 2:
				y_pred_vector = np.sum(y_pred * y_true, axis=1)
			loss = -np.log(y_pred_vector)
			loss = np.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
			return(loss)
		def backward(self, y_true,  y_pred):
			y_true = y_true.astype(int)
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
			if len(y_true.shape) == 2:
				y_true = np.argmax(y_true, axis=1)
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
		def __init__(self, gamma=1.0, beta=0.0, e=1e-7):
			self.w = gamma
			self.b = beta
			self.e = e
			self.w_m = np.zeros_like(self.w)
			self.w_c = np.zeros_like(self.w)
			self.b_m = np.zeros_like(self.b)
			self.b_c = np.zeros_like(self.b)
		def forward(self, z):
			self.z = z
			self.mean = np.mean(z, axis=0, keepdims=True)
			self.var = np.var(z, axis=0, keepdims=True)
			self.z_norm = (self.z - self.mean) / np.sqrt(self.var + self.e)
			z_new = self.w * self.z_norm + self.b
			return(z_new)
		def backward(self, dy):
			m = self.z.shape[0]
			self.dw = np.sum(dy * self.z_norm, axis=0, keepdims=True) # d_gamma
			self.db = np.sum(dy, axis=0, keepdims=True)               # d_beta
			self.dz = (self.w * (1./np.sqrt(self.var + self.e)) / m) \
			* (m*dy - np.sum(dy, axis=0) - (1./np.sqrt(self.var + self.e))**2 \
			* (self.z - self.mean) * np.sum(dy*(self.z - self.mean), axis=0))
			return(self.dz)
	#---------- Layers ----------#
	class Dense():
		''' A dense layer '''
		def __init__(self, inputs=1, outputs=1,
					alg='glorot uniform', mean=0.0, sd=0.1, a=-0.5, b=0.5,
					l1w=0, l1b=0, l2w=0, l2b=0):
			''' Initialise parameters '''
			self.l1w, self.l1b, self.l2w, self.l2b = l1w, l1b, l2w, l2b
			self.Parameters(inputs, outputs, alg, mean, sd, a, b)
		def Parameters(self, inputs, outputs, alg, mean, sd, a, b):
			''' Parameter initialisation function '''
			if alg.lower() == 'zeros':
				w = np.zeros((inputs, outputs))
			elif alg.lower() == 'ones':
				w = np.ones((inputs, outputs))
			elif alg.lower() == 'random normal':
				w = np.random.normal(loc=mean, scale=sd, size=(inputs, outputs))
			elif alg.lower() == 'random uniform':
				w = np.random.uniform(low=a, high=b, size=(inputs, outputs))
			elif alg.lower() == 'glorot normal':
				sd = 1 / (math.sqrt(inputs + outputs))
				w =  np.random.normal(loc=mean, scale=sd, size=(inputs,outputs))
			elif alg.lower() == 'glorot uniform':
				a = - math.sqrt(6) / (math.sqrt(inputs + outputs))
				b = math.sqrt(6) / (math.sqrt(inputs + outputs))
				w = np.random.uniform(low=a, high=b, size=(inputs, outputs))
			elif alg.lower() == 'he normal':
				sd = 2 / (math.sqrt(inputs + outputs))
				w = np.random.normal(loc=mean, scale=sd, size=(inputs, outputs))
			elif alg.lower() == 'he uniform':
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

	class Conv2D():
		''' A 2D convolutional layer '''
		def __init__(self, filters, kernel_size, stride=1, padding='valid',
					alg='glorot uniform', mean=0.0, sd=0.1, a=-0.5, b=0.5,
					l1w=0, l1b=0, l2w=0, l2b=0):
			''' Initialize parameters '''
			self.filters = filters
			self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
			self.stride = stride if isinstance(stride, tuple) else (stride, stride)
			self.padding = padding
			self.l1w, self.l1b, self.l2w, self.l2b = l1w, l1b, l2w, l2b
			self.Parameters(alg, mean, sd, a, b)

		def Parameters(self, alg, mean, sd, a, b):
			''' Parameter initialization function '''
			if alg.lower() == 'zeros':
				w = np.zeros((self.filters, self.kernel_size[0], self.kernel_size[1], 1))
			elif alg.lower() == 'ones':
				w = np.ones((self.filters, self.kernel_size[0], self.kernel_size[1], 1))
			elif alg.lower() == 'random normal':
				w = np.random.normal(loc=mean, scale=sd, 
					size=(self.filters, self.kernel_size[0], self.kernel_size[1], 1))
			elif alg.lower() == 'random uniform':
				w = np.random.uniform(low=a, high=b, 
					size=(self.filters, self.kernel_size[0], self.kernel_size[1], 1))
			elif alg.lower() == 'glorot normal':
				sd = 1 / (math.sqrt(self.filters + self.kernel_size[0] * self.kernel_size[1]))
				w = np.random.normal(loc=mean, scale=sd, 
					size=(self.filters, self.kernel_size[0], self.kernel_size[1], 1))
			elif alg.lower() == 'glorot uniform':
				a = -math.sqrt(6) / (math.sqrt(self.filters + self.kernel_size[0] * self.kernel_size[1]))
				b = math.sqrt(6) / (math.sqrt(self.filters + self.kernel_size[0] * self.kernel_size[1]))
				w = np.random.uniform(low=a, high=b, 
					size=(self.filters, self.kernel_size[0], self.kernel_size[1], 1))
			elif alg.lower() == 'he normal':
				sd = 2 / (math.sqrt(self.filters + self.kernel_size[0] * self.kernel_size[1]))
				w = np.random.normal(loc=mean, scale=sd, 
					size=(self.filters, self.kernel_size[0], self.kernel_size[1], 1))
			elif alg.lower() == 'he uniform':
				a = -math.sqrt(6) / (math.sqrt(self.filters))
				b = math.sqrt(6) / (math.sqrt(self.filters))
				w = np.random.uniform(low=a, high=b, 
					size=(self.filters, self.kernel_size[0], self.kernel_size[1], 1))
			self.w = w
			self.b = np.zeros((self.filters, 1))
			self.w_m = np.zeros_like(w)
			self.w_c = np.zeros_like(w)
			self.b_m = np.zeros_like(self.b)
			self.b_c = np.zeros_like(self.b)

		def forward(self, x):
			''' Forward pass '''
			self.x = x
			self.batch_size = x.shape[0]
			self.input_height = x.shape[1]
			self.input_width = x.shape[2]
			self.input_channels = x.shape[3]

			# Calculate output dimensions
			if self.padding == 'valid':
				self.output_height = (self.input_height - self.kernel_size[0]) // self.stride[0] + 1
				self.output_width = (self.input_width - self.kernel_size[1]) // self.stride[1] + 1
			else:  # 'same' padding
				self.output_height = self.input_height // self.stride[0]
				self.output_width = self.input_width // self.stride[1]

			# Initialize output
			self.output = np.zeros((self.batch_size, self.output_height, 
								  self.output_width, self.filters))

			# Perform convolution
			for i in range(self.output_height):
				for j in range(self.output_width):
					h_start = i * self.stride[0]
				 h_end = h_start + self.kernel_size[0]
					w_start = j * self.stride[1]
					w_end = w_start + self.kernel_size[1]
					
					# Extract receptive field
					receptive_field = self.x[:, h_start:h_end, w_start:w_end, :]
					
					# Convolve with each filter
					for f in range(self.filters):
						self.output[:, i, j, f] = np.sum(
							receptive_field * self.w[f], 
							axis=(1, 2, 3)
						) + self.b[f]

			# Add regularization terms
			if self.chip == 'CPU':
				self.L1w = self.l1w * np.sum(np.abs(self.w))
				self.L1b = self.l1b * np.sum(np.abs(self.b))
				self.L2w = self.l2w * np.sum(self.w**2)
				self.L2b = self.l2b * np.sum(self.b**2)
			elif self.chip == 'GPU':
				self.L1w = self.l1w * cp.sum(np.abs(self.w))
				self.L1b = self.l1b * cp.sum(np.abs(self.b))
				self.L2w = self.l2w * cp.sum(self.w**2)
				self.L2b = self.l2b * cp.sum(self.b**2)
			self.L1L2 = self.L1w + self.L1b + self.L2w + self.L2b

			return self.output

		def backward(self, dy):
			''' Backward pass '''
			# Initialize gradients
			self.dw = np.zeros_like(self.w)
			self.db = np.sum(dy, axis=(0, 1, 2), keepdims=True)
			dx = np.zeros_like(self.x)

			# Compute gradients
			for i in range(self.output_height):
				for j in range(self.output_width):
					h_start = i * self.stride[0]
					h_end = h_start + self.kernel_size[0]
					w_start = j * self.stride[1]
					w_end = w_start + self.kernel_size[1]
					
					# Extract receptive field
					receptive_field = self.x[:, h_start:h_end, w_start:w_end, :]
					
					# Update gradients
					for f in range(self.filters):
						self.dw[f] += np.sum(
							receptive_field * dy[:, i:i+1, j:j+1, f:f+1],
							axis=0
						)
						dx[:, h_start:h_end, w_start:w_end, :] += \
							self.w[f] * dy[:, i:i+1, j:j+1, f:f+1]

			# Add regularization gradients
			if self.chip == 'CPU':
				L1_dw = np.ones_like(self.w)
				L1_dw[self.w < 0] = -1
				self.dw += self.l1w * L1_dw
				L1_db = np.ones_like(self.b)
				L1_db[self.b < 0] = -1
				self.db += self.l1b * L1_db
				self.dw += 2 * self.l2w * self.w
				self.db += 2 * self.l2b * self.b
			elif self.chip == 'GPU':
				L1_dw = cp.ones_like(self.w)
				L1_dw[self.w < 0] = -1
				self.dw += self.l1w * L1_dw
				L1_db = cp.ones_like(self.b)
				L1_db[self.b < 0] = -1
				self.db += self.l1b * L1_db
				self.dw += 2 * self.l2w * self.w
				self.db += 2 * self.l2b * self.b

			return dx

	class LSTM():
		''' A Long Short-Term Memory layer '''
		def __init__(self, units, return_sequences=False,
					alg='glorot uniform', mean=0.0, sd=0.1, a=-0.5, b=0.5,
					l1w=0, l1b=0, l2w=0, l2b=0):
			''' Initialize parameters '''
			self.units = units
			self.return_sequences = return_sequences
			self.l1w, self.l1b, self.l2w, self.l2b = l1w, l1b, l2w, l2b
			self.Parameters(alg, mean, sd, a, b)

		def Parameters(self, alg, mean, sd, a, b):
			''' Parameter initialization function '''
			# Initialize weights for input gate (i), forget gate (f), cell state (c), and output gate (o)
			if alg.lower() == 'zeros':
				self.wi = np.zeros((self.units, self.units))
				self.wf = np.zeros((self.units, self.units))
				self.wc = np.zeros((self.units, self.units))
				self.wo = np.zeros((self.units, self.units))
				self.ui = np.zeros((self.units, self.units))
				self.uf = np.zeros((self.units, self.units))
				self.uc = np.zeros((self.units, self.units))
				self.uo = np.zeros((self.units, self.units))
			elif alg.lower() == 'ones':
				self.wi = np.ones((self.units, self.units))
				self.wf = np.ones((self.units, self.units))
				self.wc = np.ones((self.units, self.units))
				self.wo = np.ones((self.units, self.units))
				self.ui = np.ones((self.units, self.units))
				self.uf = np.ones((self.units, self.units))
				self.uc = np.ones((self.units, self.units))
				self.uo = np.ones((self.units, self.units))
			elif alg.lower() == 'random normal':
				self.wi = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.wf = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.wc = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.wo = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.ui = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.uf = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.uc = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.uo = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
			elif alg.lower() == 'random uniform':
				self.wi = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.wf = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.wc = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.wo = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.ui = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.uf = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.uc = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.uo = np.random.uniform(low=a, high=b, size=(self.units, self.units))
			elif alg.lower() == 'glorot normal':
				sd = 1 / (math.sqrt(self.units + self.units))
				self.wi = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.wf = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.wc = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.wo = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.ui = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.uf = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.uc = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.uo = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
			elif alg.lower() == 'glorot uniform':
				a = -math.sqrt(6) / (math.sqrt(self.units + self.units))
				b = math.sqrt(6) / (math.sqrt(self.units + self.units))
				self.wi = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.wf = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.wc = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.wo = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.ui = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.uf = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.uc = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.uo = np.random.uniform(low=a, high=b, size=(self.units, self.units))
			elif alg.lower() == 'he normal':
				sd = 2 / (math.sqrt(self.units + self.units))
				self.wi = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.wf = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.wc = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.wo = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.ui = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.uf = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.uc = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
				self.uo = np.random.normal(loc=mean, scale=sd, size=(self.units, self.units))
			elif alg.lower() == 'he uniform':
				a = -math.sqrt(6) / (math.sqrt(self.units))
				b = math.sqrt(6) / (math.sqrt(self.units))
				self.wi = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.wf = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.wc = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.wo = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.ui = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.uf = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.uc = np.random.uniform(low=a, high=b, size=(self.units, self.units))
				self.uo = np.random.uniform(low=a, high=b, size=(self.units, self.units))

			# Initialize biases
			self.bi = np.zeros((1, self.units))
			self.bf = np.zeros((1, self.units))
			self.bc = np.zeros((1, self.units))
			self.bo = np.zeros((1, self.units))

			# Initialize momentum and cache for optimization
			self.wi_m = np.zeros_like(self.wi)
			self.wi_c = np.zeros_like(self.wi)
			self.wf_m = np.zeros_like(self.wf)
			self.wf_c = np.zeros_like(self.wf)
			self.wc_m = np.zeros_like(self.wc)
			self.wc_c = np.zeros_like(self.wc)
			self.wo_m = np.zeros_like(self.wo)
			self.wo_c = np.zeros_like(self.wo)
			self.ui_m = np.zeros_like(self.ui)
			self.ui_c = np.zeros_like(self.ui)
			self.uf_m = np.zeros_like(self.uf)
			self.uf_c = np.zeros_like(self.uf)
			self.uc_m = np.zeros_like(self.uc)
			self.uc_c = np.zeros_like(self.uc)
			self.uo_m = np.zeros_like(self.uo)
			self.uo_c = np.zeros_like(self.uo)
			self.bi_m = np.zeros_like(self.bi)
			self.bi_c = np.zeros_like(self.bi)
			self.bf_m = np.zeros_like(self.bf)
			self.bf_c = np.zeros_like(self.bf)
			self.bc_m = np.zeros_like(self.bc)
			self.bc_c = np.zeros_like(self.bc)
			self.bo_m = np.zeros_like(self.bo)
			self.bo_c = np.zeros_like(self.bo)

		def forward(self, x):
			''' Forward pass '''
			self.x = x
			self.batch_size = x.shape[0]
			self.sequence_length = x.shape[1]
			self.input_features = x.shape[2]

			# Initialize hidden state and cell state
			self.h = np.zeros((self.batch_size, self.sequence_length, self.units))
			self.c = np.zeros((self.batch_size, self.sequence_length, self.units))

			# Initialize gates
			self.i = np.zeros((self.batch_size, self.sequence_length, self.units))
			self.f = np.zeros((self.batch_size, self.sequence_length, self.units))
			self.c_bar = np.zeros((self.batch_size, self.sequence_length, self.units))
			self.o = np.zeros((self.batch_size, self.sequence_length, self.units))

			# Process each time step
			for t in range(self.sequence_length):
				# Input gate
				self.i[:, t] = self.sigmoid(np.dot(x[:, t], self.wi.T) + 
										  np.dot(self.h[:, t-1], self.ui.T) + self.bi)
				
				# Forget gate
				self.f[:, t] = self.sigmoid(np.dot(x[:, t], self.wf.T) + 
										  np.dot(self.h[:, t-1], self.uf.T) + self.bf)
				
				# Cell state candidate
				self.c_bar[:, t] = np.tanh(np.dot(x[:, t], self.wc.T) + 
										 np.dot(self.h[:, t-1], self.uc.T) + self.bc)
				
				# Update cell state
				self.c[:, t] = self.f[:, t] * self.c[:, t-1] + self.i[:, t] * self.c_bar[:, t]
				
				# Output gate
				self.o[:, t] = self.sigmoid(np.dot(x[:, t], self.wo.T) + 
										  np.dot(self.h[:, t-1], self.uo.T) + self.bo)
				
				# Update hidden state
				self.h[:, t] = self.o[:, t] * np.tanh(self.c[:, t])

			# Add regularization terms
			if self.chip == 'CPU':
				self.L1w = self.l1w * (np.sum(np.abs(self.wi)) + np.sum(np.abs(self.wf)) + 
									 np.sum(np.abs(self.wc)) + np.sum(np.abs(self.wo)) +
									 np.sum(np.abs(self.ui)) + np.sum(np.abs(self.uf)) + 
									 np.sum(np.abs(self.uc)) + np.sum(np.abs(self.uo)))
				self.L1b = self.l1b * (np.sum(np.abs(self.bi)) + np.sum(np.abs(self.bf)) + 
									 np.sum(np.abs(self.bc)) + np.sum(np.abs(self.bo)))
				self.L2w = self.l2w * (np.sum(self.wi**2) + np.sum(self.wf**2) + 
									 np.sum(self.wc**2) + np.sum(self.wo**2) +
									 np.sum(self.ui**2) + np.sum(self.uf**2) + 
									 np.sum(self.uc**2) + np.sum(self.uo**2))
				self.L2b = self.l2b * (np.sum(self.bi**2) + np.sum(self.bf**2) + 
									 np.sum(self.bc**2) + np.sum(self.bo**2))
			elif self.chip == 'GPU':
				self.L1w = self.l1w * (cp.sum(np.abs(self.wi)) + cp.sum(np.abs(self.wf)) + 
									 cp.sum(np.abs(self.wc)) + cp.sum(np.abs(self.wo)) +
									 cp.sum(np.abs(self.ui)) + cp.sum(np.abs(self.uf)) + 
									 cp.sum(np.abs(self.uc)) + cp.sum(np.abs(self.uo)))
				self.L1b = self.l1b * (cp.sum(np.abs(self.bi)) + cp.sum(np.abs(self.bf)) + 
									 cp.sum(np.abs(self.bc)) + cp.sum(np.abs(self.bo)))
				self.L2w = self.l2w * (cp.sum(self.wi**2) + cp.sum(self.wf**2) + 
									 cp.sum(self.wc**2) + cp.sum(self.wo**2) +
									 cp.sum(self.ui**2) + cp.sum(self.uf**2) + 
									 cp.sum(self.uc**2) + cp.sum(self.uo**2))
				self.L2b = self.l2b * (cp.sum(self.bi**2) + cp.sum(self.bf**2) + 
									 cp.sum(self.bc**2) + cp.sum(self.bo**2))
			self.L1L2 = self.L1w + self.L1b + self.L2w + self.L2b

			if self.return_sequences:
				return self.h
			else:
				return self.h[:, -1]

		def backward(self, dy):
			''' Backward pass '''
			# Initialize gradients
			self.dwi = np.zeros_like(self.wi)
			self.dwf = np.zeros_like(self.wf)
			self.dwc = np.zeros_like(self.wc)
			self.dwo = np.zeros_like(self.wo)
			self.dui = np.zeros_like(self.ui)
			self.duf = np.zeros_like(self.uf)
			self.duc = np.zeros_like(self.uc)
			self.duo = np.zeros_like(self.uo)
			self.dbi = np.zeros_like(self.bi)
			self.dbf = np.zeros_like(self.bf)
			self.dbc = np.zeros_like(self.bc)
			self.dbo = np.zeros_like(self.bo)
			dx = np.zeros_like(self.x)

			# Initialize cell state and hidden state gradients
			dc_next = np.zeros((self.batch_size, self.units))
			dh_next = np.zeros((self.batch_size, self.units))

			# Backpropagate through time
			for t in reversed(range(self.sequence_length)):
				if self.return_sequences:
					dh = dy[:, t] + dh_next
				else:
					dh = dy if t == self.sequence_length - 1 else dh_next

				# Output gate gradients
				do = dh * np.tanh(self.c[:, t])
				do = do * self.sigmoid(self.o[:, t]) * (1 - self.o[:, t])
				self.dwo += np.dot(self.x[:, t].T, do)
				self.duo += np.dot(self.h[:, t-1].T, do)
				self.dbo += np.sum(do, axis=0, keepdims=True)

				# Cell state gradients
				dc = dh * self.o[:, t] * (1 - np.tanh(self.c[:, t])**2)
				dc = dc + dc_next

				# Cell state candidate gradients
				dc_bar = dc * self.i[:, t]
				dc_bar = dc_bar * (1 - self.c_bar[:, t]**2)
				self.dwc += np.dot(self.x[:, t].T, dc_bar)
				self.duc += np.dot(self.h[:, t-1].T, dc_bar)
				self.dbc += np.sum(dc_bar, axis=0, keepdims=True)

				# Input gate gradients
				di = dc * self.c_bar[:, t]
				di = di * self.sigmoid(self.i[:, t]) * (1 - self.i[:, t])
				self.dwi += np.dot(self.x[:, t].T, di)
				self.dui += np.dot(self.h[:, t-1].T, di)
				self.dbi += np.sum(di, axis=0, keepdims=True)

				# Forget gate gradients
				df = dc * self.c[:, t-1]
				df = df * self.sigmoid(self.f[:, t]) * (1 - self.f[:, t])
				self.dwf += np.dot(self.x[:, t].T, df)
				self.duf += np.dot(self.h[:, t-1].T, df)
				self.dbf += np.sum(df, axis=0, keepdims=True)

				# Update gradients for next time step
				dc_next = dc * self.f[:, t]
				dh_next = np.dot(do, self.uo) + np.dot(dc_bar, self.uc) + \
						  np.dot(di, self.ui) + np.dot(df, self.uf)

				# Input gradients
				dx[:, t] = np.dot(do, self.wo) + np.dot(dc_bar, self.wc) + \
						   np.dot(di, self.wi) + np.dot(df, self.wf)

			# Add regularization gradients
			if self.chip == 'CPU':
				# L1 regularization
				L1_dwi = np.ones_like(self.wi)
				L1_dwi[self.wi < 0] = -1
				self.dwi += self.l1w * L1_dwi
				L1_dwf = np.ones_like(self.wf)
				L1_dwf[self.wf < 0] = -1
				self.dwf += self.l1w * L1_dwf
				L1_dwc = np.ones_like(self.wc)
				L1_dwc[self.wc < 0] = -1
				self.dwc += self.l1w * L1_dwc
				L1_dwo = np.ones_like(self.wo)
				L1_dwo[self.wo < 0] = -1
				self.dwo += self.l1w * L1_dwo
				L1_dui = np.ones_like(self.ui)
				L1_dui[self.ui < 0] = -1
				self.dui += self.l1w * L1_dui
				L1_duf = np.ones_like(self.uf)
				L1_duf[self.uf < 0] = -1
				self.duf += self.l1w * L1_duf
				L1_duc = np.ones_like(self.uc)
				L1_duc[self.uc < 0] = -1
				self.duc += self.l1w * L1_duc
				L1_duo = np.ones_like(self.uo)
				L1_duo[self.uo < 0] = -1
				self.duo += self.l1w * L1_duo
				L1_dbi = np.ones_like(self.bi)
				L1_dbi[self.bi < 0] = -1
				self.dbi += self.l1b * L1_dbi
				L1_dbf = np.ones_like(self.bf)
				L1_dbf[self.bf < 0] = -1
				self.dbf += self.l1b * L1_dbf
				L1_dbc = np.ones_like(self.bc)
				L1_dbc[self.bc < 0] = -1
				self.dbc += self.l1b * L1_dbc
				L1_dbo = np.ones_like(self.bo)
				L1_dbo[self.bo < 0] = -1
				self.dbo += self.l1b * L1_dbo

				# L2 regularization
				self.dwi += 2 * self.l2w * self.wi
				self.dwf += 2 * self.l2w * self.wf
				self.dwc += 2 * self.l2w * self.wc
				self.dwo += 2 * self.l2w * self.wo
				self.dui += 2 * self.l2w * self.ui
				self.duf += 2 * self.l2w * self.uf
				self.duc += 2 * self.l2w * self.uc
				self.duo += 2 * self.l2w * self.uo
				self.dbi += 2 * self.l2b * self.bi
				self.dbf += 2 * self.l2b * self.bf
				self.dbc += 2 * self.l2b * self.bc
				self.dbo += 2 * self.l2b * self.bo
			elif self.chip == 'GPU':
				# L1 regularization
				L1_dwi = cp.ones_like(self.wi)
				L1_dwi[self.wi < 0] = -1
				self.dwi += self.l1w * L1_dwi
				L1_dwf = cp.ones_like(self.wf)
				L1_dwf[self.wf < 0] = -1
				self.dwf += self.l1w * L1_dwf
				L1_dwc = cp.ones_like(self.wc)
				L1_dwc[self.wc < 0] = -1
				self.dwc += self.l1w * L1_dwc
				L1_dwo = cp.ones_like(self.wo)
				L1_dwo[self.wo < 0] = -1
				self.dwo += self.l1w * L1_dwo
				L1_dui = cp.ones_like(self.ui)
				L1_dui[self.ui < 0] = -1
				self.dui += self.l1w * L1_dui
				L1_duf = cp.ones_like(self.uf)
				L1_duf[self.uf < 0] = -1
				self.duf += self.l1w * L1_duf
				L1_duc = cp.ones_like(self.uc)
				L1_duc[self.uc < 0] = -1
				self.duc += self.l1w * L1_duc
				L1_duo = cp.ones_like(self.uo)
				L1_duo[self.uo < 0] = -1
				self.duo += self.l1w * L1_duo
				L1_dbi = cp.ones_like(self.bi)
				L1_dbi[self.bi < 0] = -1
				self.dbi += self.l1b * L1_dbi
				L1_dbf = cp.ones_like(self.bf)
				L1_dbf[self.bf < 0] = -1
				self.dbf += self.l1b * L1_dbf
				L1_dbc = cp.ones_like(self.bc)
				L1_dbc[self.bc < 0] = -1
				self.dbc += self.l1b * L1_dbc
				L1_dbo = cp.ones_like(self.bo)
				L1_dbo[self.bo < 0] = -1
				self.dbo += self.l1b * L1_dbo

				# L2 regularization
				self.dwi += 2 * self.l2w * self.wi
				self.dwf += 2 * self.l2w * self.wf
				self.dwc += 2 * self.l2w * self.wc
				self.dwo += 2 * self.l2w * self.wo
				self.dui += 2 * self.l2w * self.ui
				self.duf += 2 * self.l2w * self.uf
				self.duc += 2 * self.l2w * self.uc
				self.duo += 2 * self.l2w * self.uo
				self.dbi += 2 * self.l2b * self.bi
				self.dbf += 2 * self.l2b * self.bf
				self.dbc += 2 * self.l2b * self.bc
				self.dbo += 2 * self.l2b * self.bo

			return dx

		def sigmoid(self, x):
			''' Sigmoid activation function '''
			return 1 / (1 + np.exp(-x))

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
		else: steps = 1
		if early_stop: STOP = self.EarlyStopping()
		args = {'cost_batch':None, 'accuracy_batch':None,
				'cost_train':None, 'accuracy_train':None,
				'cost_valid':None, 'accuracy_valid':None,
				'cost_tests':None, 'accuracy_tests':None,
				'epoch':None, 'epochs':epochs, 'step':None, 'steps':steps,
				'time':None}
		for epoch in range(epochs):
			args['epoch'] = epoch + 1
			Estart = time.process_time()
			# Shuffle training datatset
			X_train, Y_train = self.shuffle_data(X_train, Y_train)
			for step in range(steps):
				args['step'] = step + 1
				Bstart = time.process_time()
				# Batch segmentation
				if batch_size is not None:
					start_idx = step * batch_size
					end_idx = min((step + 1) * batch_size, len(X_train))
					X_batch = X_train[start_idx:end_idx]
					Y_batch = Y_train[start_idx:end_idx]
				else:
					X_batch = X_train
					Y_batch = Y_train
				# Forward propagation
				y_true = Y_batch
				y_pred = self.forward(X_batch)
				L1L2 = [l.L1L2 for l in self.layers if isinstance(l,self.Dense)]
				L1L2 = sum(L1L2)
				cost_batch = self.cost(loss_fn.forward(y_true, y_pred)) + L1L2
				accuracy_batch = acc.calc(y_true, y_pred)
				# Backpropagation
				dy = loss_fn.backward(y_true, y_pred)
				dx = self.backward(dy)
				# Gradient descent
				for layer in self.layers:
					if isinstance(layer, (self.Dense, self.BatchNorm)):
						if optimiser.lower() == 'sgd':
							self.SGD(lr, decay, epoch, layer)
						elif optimiser.lower() == 'adagrad':
							self.Adagrad(lr, decay, epoch, e, layer)
						elif optimiser.lower() == 'rmsprop':
							self.RMSprop(lr, decay, epoch, beta1, e, layer)
						elif optimiser.lower() == 'adam':
							self.Adam(lr, decay, epoch, beta1, beta2, e, layer)
				Bend = time.process_time()
				args['cost_batch'] = cost_batch
				args['accuracy_batch'] = accuracy_batch
				args['time'] = Bend - Bstart
				if verbose == 2: self.verbosity('Batch', args)
			# Evaluate training set
			y_true = Y_train
			y_pred = self.predict(X_train)
			L1L2 = [l.L1L2 for l in self.layers if isinstance(l,self.Dense)]
			L1L2 = sum(L1L2)
			cost_train = self.cost(loss_fn.forward(y_true, y_pred)) + L1L2
			accuracy_train = acc.calc(y_true, y_pred)
			Eend = time.process_time()
			args['cost_train'] = cost_train
			args['accuracy_train'] = accuracy_train
			args['time'] = Eend - Estart
			if (verbose == 1 or verbose == 2) and (X_valid is None):
				self.verbosity('Train', args)
			# Evaluate validation set
			if X_valid is not None and Y_valid is not None:
				Vstart = time.process_time()
				y_true = Y_valid
				y_pred = self.predict(X_valid)
				L1L2 = [l.L1L2 for l in self.layers if isinstance(l,self.Dense)]
				L1L2 = sum(L1L2)
				cost_valid = self.cost(loss_fn.forward(y_true, y_pred)) + L1L2
				accuracy_valid = acc.calc(y_true, y_pred)
				Vend = time.process_time()
				args['cost_valid'] = cost_valid
				args['accuracy_valid'] = accuracy_valid
				args['time'] = Vend - Vstart
				if verbose == 1 or verbose == 2: self.verbosity('Train', args)
			# Early stop checkpoint
			if early_stop and STOP.check(epoch, cost_train): break
		# Evaluate test set
		if X_tests is not None and Y_tests is not None:
			Tstart = time.process_time()
			y_true = Y_tests
			y_pred = self.predict(X_tests)
			L1L2 = [l.L1L2 for l in self.layers if isinstance(l,self.Dense)]
			L1L2 = sum(L1L2)
			cost_tests = self.cost(loss_fn.forward(y_true, y_pred)) + L1L2
			accuracy_tests = acc.calc(y_true, y_pred)
			Tend = time.process_time()
			args['cost_tests'] = cost_tests
			args['accuracy_tests'] = accuracy_tests
			args['time'] = Tend - Tstart
			if verbose == 1 or verbose == 2: self.verbosity('Tests', args)
