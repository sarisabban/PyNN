import time
import math
import pickle
import numpy as np

try: import cupy as np
except: pass

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
		self.track = (
			self.Dense,
			self.Reshape,
			self.Flatten,
			self.Pool,
			self.Conv)
	def add(self, layer):
		''' Add a layer to the network '''
		if isinstance(layer, self.track):
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
				print(f'\033[31mEarly Stop: Training loss plateaued\033[0m')
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
			if isinstance(layer, self.track):
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
	def ParamInit(self, shape, alg, mean, sd, a, b):
		''' Parameter initialisation function '''
		if isinstance(shape, int):
			shape = (shape,)
		if len(shape) == 1:
			# 1D shape (e.g., bias): only allow simple initializations
			if alg.lower() == 'zeros':
				w = np.zeros(shape)
			elif alg.lower() == 'ones':
				w = np.ones(shape)
			elif alg.lower() == 'random normal':
				w = np.random.normal(loc=mean, scale=sd, size=shape)
			elif alg.lower() == 'random uniform':
				w = np.random.uniform(low=a, high=b, size=shape)
			elif alg.lower() == 'integers':
				w = np.random.randint(low=a, high=b, size=shape)
			else:
				raise ValueError(f'Unknown initialization algorithm for 1D shape: {alg}')
			return w
		elif len(shape) == 2:
			inputs, outputs = shape
		else:
			receptive_field_size = np.prod(shape[2:])
			inputs = shape[1] * receptive_field_size
			outputs = shape[0] * receptive_field_size
		if alg.lower() == 'zeros':
			w = np.zeros(shape)
		elif alg.lower() == 'ones':
			w = np.ones(shape)
		elif alg.lower() == 'random normal':
			w = np.random.normal(loc=mean, scale=sd, size=shape)
		elif alg.lower() == 'random uniform':
			w = np.random.uniform(low=a, high=b, size=shape)
		elif alg.lower() == 'glorot normal':
			sd = math.sqrt(2 / (inputs + outputs))
			w =  np.random.normal(loc=mean, scale=sd, size=shape)
		elif alg.lower() == 'glorot uniform':
			a = - math.sqrt(6) / (math.sqrt(inputs + outputs))
			b = math.sqrt(6) / (math.sqrt(inputs + outputs))
			w = np.random.uniform(low=a, high=b, size=shape)
		elif alg.lower() == 'he normal':
			sd = math.sqrt(2 / inputs)
			w = np.random.normal(loc=mean, scale=sd, size=shape)
		elif alg.lower() == 'he uniform':
			a = - math.sqrt(6) / (math.sqrt(inputs))
			b = math.sqrt(6) / (math.sqrt(inputs))
			w = np.random.uniform(low=a, high=b, size=shape)
		elif alg.lower() == 'integers':
			w = np.random.randint(low=a, high=b, size=shape)
		else:
			raise ValueError(f'Unknown initialization algorithm: {alg}')
		return(w)
	def Padding(self, x, kernel=(3,3), stride=(1,1), val='zeros', alg='valid'):
		''' A padding function '''
		mode = {'zeros':'constant', 'replicate':'edge', 'reflect':'wrap'}
		ndim = x.ndim
		if isinstance(kernel, int): kernel = (kernel,) * ndim
		if isinstance(stride, int): stride = (stride,) * ndim
		if alg.lower() == 'valid':
			width = [(0, 0) for _ in range(ndim)]
		elif alg.lower() == 'same':
			width = []
			for i, k, s in zip(x.shape, kernel, stride):
				pad_total = max((i - 1) * s + k - i, 0)
				pad_before = pad_total // 2
				pad_after = pad_total - pad_before
				width.append((pad_before, pad_after))
			# Add padding for channel dimension if needed
			if x.ndim > len(width):
				width.append((0, 0))
		elif alg.lower() == 'full':
			width = []
			for i, k, s in zip(x.shape, kernel, stride):
				pad_total = max((i - 1) * s + k - i, 0)+2
				pad_before = pad_total // 2
				pad_after = pad_total - pad_before
				width.append((pad_before, pad_after))
			# Add padding for channel dimension if needed
			if x.ndim > len(width):
				width.append((0, 0))
		y = np.pad(x, width, mode[val])
		return(y)
	#---------- Activation Functions ----------#
	class Step():
		''' The Step activation function (for binary classification) '''
		def forward(self, z):
			y = np.where(z >= 0, 1, 0)
			return(y)
		def backward(self, dy):
			dz = np.zeros_like(dy)
			return(dz)
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
	class Flatten():
		''' Flattens a layer to 1D '''
		def forward(self, x):
			self.input_shape = x.shape
			new_x = x.reshape(x.shape[0], -1)
			return(new_x)
		def backward(self, dz):
			new_dz = dz.reshape(self.input_shape)
			return(new_dz)
	class Reshape():
		''' Reshape a layer '''
		def __init__(self, input_shape, output_shape):
			self.input_shape = input_shape
			self.output_shape = output_shape
		def forward(self, x):
			new_x = np.reshape(x, self.output_shape)
			return(new_x)
		def backward(self, dz):
			new_dz = np.reshape(dz, self.input_shape)
			return(new_dz)
	class Pool():
		''' An n dimensional pooling layer '''
		def __init__(self, window=(2, 2), stride=(2, 2), alg='max'):
			if isinstance(window, int): self.window = (window,)
			else: self.window = window
			if isinstance(stride, int): self.stride = (stride,)
			else: self.stride = stride
			self.alg = alg.lower()
		def forward(self, x):
			self.x = x
			d, w, s = x.shape, self.window, self.stride
			stds = x.strides
			output_shape = tuple((d[i]-w[i]) // s[i] + 1 for i in range(len(d)))
			window_shape = output_shape + self.window
			window_strides = tuple(stds[i] * s[i] for i in range(len(d))) + stds
			self.windows = np.lib.stride_tricks.as_strided(
			x, shape=window_shape, strides=window_strides)
			n = len(x.shape)
			axis = tuple(range(n, n + n))
			if   self.alg == 'max': output = np.max(self.windows,  axis=axis)
			elif self.alg == 'avg': output = np.mean(self.windows, axis=axis)
			return(output)
		def backward(self, dz):
			sh, st, wn = dz.shape, self.stride, self.window
			if self.alg == 'max':
				dx = np.zeros_like(self.x)
				idx = np.argmax(self.windows.reshape(dz.shape + (-1,)), axis=-1)
				coord = np.unravel_index(idx, self.window)
				mesh = np.meshgrid(*[np.arange(s) for s in sh], indexing='ij')
				idxs = [mesh[i] * st[i] + coord[i] for i in range(len(st))]
				np.add.at(dx, tuple(idxs), dz)
			elif self.alg == 'avg':
				dx = np.zeros_like(self.x, dtype=np.float64)
				for idx in np.ndindex(sh):
					slc = tuple(slice(i*s,i*s+w) for i,s,w in zip(idx,st,wn))
					dx[slc] += dz[idx] / np.prod(self.window)
			return(dx)
	class Dense():
		''' A dense layer '''
		def __init__(self, inputs=1, outputs=1,
					alg='glorot uniform', mean=0.0, sd=0.1, a=-0.5, b=0.5,
					l1w=0, l1b=0, l2w=0, l2b=0):
			self.w = PyNN.ParamInit(PyNN, (inputs,outputs), alg, mean, sd, a, b)
			self.l1w, self.l1b, self.l2w, self.l2b = l1w, l1b, l2w, l2b
			self.b = np.zeros((1, outputs))
			self.beta = np.zeros((1, outputs))
			self.gamma = np.ones((1, outputs))
			self.w_m = np.zeros_like(self.w)
			self.w_c = np.zeros_like(self.w)
			self.b_m = np.zeros_like(self.b)
			self.b_c = np.zeros_like(self.b)
		def forward(self, x):
			self.x = x
			self.L1w = self.l1w * np.sum(np.abs(self.w))
			self.L1b = self.l1b * np.sum(np.abs(self.b))
			self.L2w = self.l2w * np.sum(self.w**2)
			self.L2b = self.l2b * np.sum(self.b**2)
			z = np.dot(self.x, self.w) + self.b
			self.L1L2 = self.L1w + self.L1b + self.L2w + self.L2b
			return(z)
		def backward(self, dz):
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
			return(self.dx)

class Conv():
	''' A convolution layer supporting 1D, 2D, and 3D '''
	def __init__(self, input_shape, kernel_shape, kernel_number=1, stride_shape=1,
				 alg='random uniform', mean=0.0, sd=0.1, a=-0.5, b=0.5, padding='valid',
				 l1w=0, l1b=0, l2w=0, l2b=0):
		self.padding = padding
		if isinstance(input_shape, int):
			input_shape = (input_shape,)
		if isinstance(kernel_shape, int):
			kernel_shape = (kernel_shape,)
		if isinstance(stride_shape, int):
			stride_shape = (stride_shape,) * len(input_shape)
		self.input_shape = input_shape
		self.kernel_shape = kernel_shape
		self.kernel_number = kernel_number
		self.stride_shape = stride_shape
		if   len(input_shape) == 1: in_channels = 1
		elif len(input_shape) == 2: in_channels = 1
		elif len(input_shape) == 3: in_channels = 1
		else: raise ValueError('Only 1D, 2D, 3D supported')
		self.K_shape = tuple(kernel_shape) + (in_channels, kernel_number)
		self.B_shape = (kernel_number,)
		self.K = PyNN.ParamInit(PyNN, self.K_shape, alg, mean, sd, a, b)
		self.B = PyNN.ParamInit(PyNN, self.B_shape, alg, mean, sd, a, b)
		self.l1w, self.l1b, self.l2w, self.l2b = l1w, l1b, l2w, l2b
		self.w_m = np.zeros_like(self.K)
		self.w_c = np.zeros_like(self.K)
		self.b_m = np.zeros_like(self.B)
		self.b_c = np.zeros_like(self.B)
		# Store original shapes for backward pass
		self.original_K_shape = self.K.shape
		self.original_B_shape = self.B.shape
	def forward(self, x):
		self.x = x
		ndim = x.ndim
		if ndim == len(self.input_shape):
			x = x[..., np.newaxis]
		if self.padding == 'same':
			x = PyNN.Padding(PyNN, x, kernel=self.kernel_shape, stride=self.stride_shape, val='zeros', alg='same')
		windows = np.lib.stride_tricks.sliding_window_view(x, self.kernel_shape, axis=tuple(range(len(self.kernel_shape))))
		strided_slices = tuple(slice(None, None, s) for s in self.stride_shape)
		windows = windows[strided_slices]
		if len(self.input_shape) == 1:
			self.y = np.einsum('ijk,jkl->il', windows, self.K)
		elif len(self.input_shape) == 2:
			self.y = np.einsum('hwijc,ijco->hwo', windows, self.K)
		elif len(self.input_shape) == 3:
			w = np.squeeze(windows)
			K = np.squeeze(self.K)
			if   w.ndim == 5 and K.ndim == 4:
				self.y = np.einsum('ijklm,klmn->ijn', w, K)
			elif w.ndim == 6 and K.ndim == 4:
				self.y = np.einsum('abcdef,defg->abcg', w, K)
			else: raise ValueError('Unexpected shape for 3D convolution einsum')
		else: raise ValueError('Only 1D, 2D, 3D supported')
		self.y += self.B
		self.y = self.y.squeeze()
		# Calculate L1L2 regularization
		self.L1w = self.l1w * np.sum(np.abs(self.K))
		self.L1b = self.l1b * np.sum(np.abs(self.B))
		self.L2w = self.l2w * np.sum(self.K**2)
		self.L2b = self.l2b * np.sum(self.B**2)
		self.L1L2 = self.L1w + self.L1b + self.L2w + self.L2b
		return(self.y)
	def backward(self, dz):
		''' Backward pass for convolution layer '''
		# Initialize gradients
		self.dK = np.zeros_like(self.K)
		self.dB = np.sum(dz, axis=tuple(range(len(self.input_shape))))
		self.dx = np.zeros_like(self.x)

		# Add channel dimension if needed
		if dz.ndim == len(self.input_shape):
			dz = dz[..., np.newaxis]

		# Get padded input if needed
		x = self.x
		if self.padding == 'same':
			x = PyNN.Padding(PyNN, x, kernel=self.kernel_shape, stride=self.stride_shape, val='zeros', alg='same')

		# Create sliding windows for input
		windows = np.lib.stride_tricks.sliding_window_view(x, self.kernel_shape, axis=tuple(range(len(self.kernel_shape))))
		strided_slices = tuple(slice(None, None, s) for s in self.stride_shape)
		windows = windows[strided_slices]

		# Compute gradients based on input dimensions
		if len(self.input_shape) == 1:
			# 1D convolution
			# dK: (k, in_ch, out_ch) -- here in_ch=1
			self.dK = np.einsum('oi,oj->ij', windows, dz)
			self.dK = self.dK[:, np.newaxis, :]
			# dx: (in_size, in_ch)
			if self.dx.ndim == 1:
				self.dx = self.dx[:, np.newaxis]
			for i in range(windows.shape[0]):
				start_idx = i * self.stride_shape[0]
				end_idx = start_idx + self.kernel_shape[0]
				# dz[i]: (out_ch,), self.K: (k, in_ch, out_ch)
				# We want to sum over out_ch, result: (k, in_ch)
				self.dx[start_idx:end_idx, :] += np.einsum('l,klm->km', dz[i].ravel(), self.K)

		elif len(self.input_shape) == 2:
			# 2D convolution
			# Add in_channel dimension if missing
			if windows.ndim == 4:
				windows = windows[..., np.newaxis]
			# dK: (kh, kw, in_ch, out_ch)
			self.dK = np.einsum('hwijc,hwo->ijco', windows, dz)
			# dx: (h, w, in_ch)
			if self.dx.ndim == 2:
				self.dx = self.dx[..., np.newaxis]
			for i in range(windows.shape[0]):
				for j in range(windows.shape[1]):
					h_start = i * self.stride_shape[0]
					h_end = h_start + self.kernel_shape[0]
					w_start = j * self.stride_shape[1]
					w_end = w_start + self.kernel_shape[1]
					# Handle edge cases for padding
					dx_slice = self.dx[h_start:h_end, w_start:w_end, :]
					k_h = dx_slice.shape[0]
					k_w = dx_slice.shape[1]
					self.dx[h_start:h_end, w_start:w_end, :] += np.einsum('o,ijco->ijc', dz[i, j], self.K[:k_h, :k_w, :, :])

		elif len(self.input_shape) == 3:
			# 3D convolution
			w = windows
			K = self.K
			# Ensure w and K have shape (out_d1, out_d2, out_d3, k1, k2, k3, in_ch)
			if w.ndim == 6:
				w = np.expand_dims(w, -1)  # add in_ch=1 if missing
			if K.ndim == 4:
				K = np.expand_dims(K, -2)  # add in_ch=1 if missing
			# dK: (k1, k2, k3, in_ch, out_ch)
			self.dK = np.einsum('abcxyzl,abcg->xyzlg', w, dz)
			# dx: (d1, d2, d3, in_ch)
			if self.dx.ndim == 3:
				self.dx = self.dx[..., np.newaxis]
			for i in range(w.shape[0]):
				for j in range(w.shape[1]):
					for k in range(w.shape[2]):
						d1_start = i * self.stride_shape[0]
						d1_end = d1_start + self.kernel_shape[0]
						d2_start = j * self.stride_shape[1]
						d2_end = d2_start + self.kernel_shape[1]
						d3_start = k * self.stride_shape[2]
						d3_end = d3_start + self.kernel_shape[2]
						self.dx[d1_start:d1_end, d2_start:d2_end, d3_start:d3_end, :] += np.einsum('g,xyzlg->xyzl', dz[i, j, k], K)

		# Add L1L2 regularization gradients
		L1_dK = np.ones_like(self.K)
		L1_dK[self.K < 0] = -1
		self.dK += self.l1w * L1_dK
		L1_dB = np.ones_like(self.B)
		L1_dB[self.B < 0] = -1
		self.dB += self.l1b * L1_dB
		self.dK += 2 * self.l2w * self.K
		self.dB += 2 * self.l2b * self.B

		# Ensure gradients have correct shapes
		self.dK = self.dK.reshape(self.original_K_shape)
		self.dB = self.dB.reshape(self.original_B_shape)

		# Remove padding from dx if needed
		if self.padding == 'same':
			# Calculate padding amounts
			pad_width = []
			for i, k, s in zip(self.x.shape, self.kernel_shape, self.stride_shape):
				pad_total = max((i - 1) * s + k - i, 0)
				pad_before = pad_total // 2
				pad_after = pad_total - pad_before
				pad_width.append((pad_before, pad_after))
			# Add padding for channel dimension if needed
			if self.x.ndim > len(pad_width):
				pad_width.append((0, 0))
			# Remove padding
			slices = tuple(slice(p[0], -p[1] if p[1] > 0 else None) for p in pad_width)
			self.dx = self.dx[slices]

		return self.dx




# --- CNN Layer Test Cases ---
print("\n--- 1D Conv Test ---")
x1d = np.arange(9)
C1d = Conv(input_shape=9, kernel_shape=3, kernel_number=1, stride_shape=1, padding='valid', alg='integers', a=0, b=5)
y1d = C1d.forward(x1d)
print("1D output shape:", y1d.shape)
# Test backward pass
dy1d = np.ones_like(y1d)
dx1d = C1d.backward(dy1d)
print("1D gradients:")
print("dK shape:", C1d.dK.shape)
print("dB shape:", C1d.dB.shape)
print("dx shape:", dx1d.shape)

print("\n--- 2D Conv Test ---")
x2d = np.arange(81).reshape(9,9)
C2d = Conv(input_shape=(9,9), kernel_shape=(2,2), kernel_number=9, stride_shape=(2,2), padding='valid', alg='integers', a=0, b=5)
y2d = C2d.forward(x2d)
print("2D output shape:", y2d.shape)
# Test backward pass
dy2d = np.ones_like(y2d)
dx2d = C2d.backward(dy2d)
print("2D gradients:")
print("dK shape:", C2d.dK.shape)
print("dB shape:", C2d.dB.shape)
print("dx shape:", dx2d.shape)

print("\n--- 3D Conv Test ---")
x3d = np.arange(12*12*12).reshape(12,12,12)
C3d = Conv(input_shape=(12,12,12), kernel_shape=(2,2,2), kernel_number=5, stride_shape=(2,2,2), padding='valid', alg='integers', a=0, b=5)
y3d = C3d.forward(x3d)
print("3D output shape:", y3d.shape)
# Test backward pass
dy3d = np.ones_like(y3d)
dx3d = C3d.backward(dy3d)
print("3D gradients:")
print("dK shape:", C3d.dK.shape)
print("dB shape:", C3d.dB.shape)
print("dx shape:", dx3d.shape)

# Test with 'same' padding
print("\n--- 2D Conv Test (same padding) ---")
x2d_same = np.arange(81).reshape(9,9)
C2d_same = Conv(input_shape=(9,9), kernel_shape=(3,3), kernel_number=9, stride_shape=(1,1), padding='same', alg='integers', a=0, b=5)
y2d_same = C2d_same.forward(x2d_same)
print("2D output shape (same padding):", y2d_same.shape)
# Test backward pass
dy2d_same = np.ones_like(y2d_same)
dx2d_same = C2d_same.backward(dy2d_same)
print("2D gradients (same padding):")
print("dK shape:", C2d_same.dK.shape)
print("dB shape:", C2d_same.dB.shape)
print("dx shape:", dx2d_same.shape)
