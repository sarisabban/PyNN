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
			self.Pool)
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
		elif alg.lower() == 'full':
			width = []
			for i, k, s in zip(x.shape, kernel, stride):
				pad_total = max((i - 1) * s + k - i, 0)+2
				pad_before = pad_total // 2
				pad_after = pad_total - pad_before
				width.append((pad_before, pad_after))
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




"""
https://www.youtube.com/watch?v=Lakz2MoHy6o
[ ] 1D
[ ] 2D
[ ] 3D
[ ] L1L2
[ ] Adam optimiser weights
[ ] Add self.CNN to line 573 in def.train() and def.show()
"""

class Conv():
	''' A convolution layer '''
	def __init__(self, input_shape=(3, 3),
				kernel_shape=(2, 2), kernel_number=1,
				stride_shape=(1, 1),
				alg='random uniform', mean=0.0, sd=0.1, a=-0.5, b=0.5,
				padding='valid',
				#l1w=0, l1b=0, l2w=0, l2b=0
				):
		self.padding = padding
		self.Is = input_shape
		self.Ks = kernel_shape
		self.Kn = kernel_number
		self.Ss = stride_shape
		if padding == 'valid':
			if isinstance(self.Is, int):
				self.Bs = ((self.Is - self.Ks)//self.Ss) + 1
			elif isinstance(self.Is, tuple):
				I, K, S = self.Is, self.Ks, self.Ss
				self.Bs = tuple(((i - k)//s) + 1 for i, k, s in zip(I, K, S))
		elif padding == 'same':
			self.Bs = input_shape
		if self.Kn > 1:
			if isinstance(self.Is, int):
				self.Ks = (self.Ks, self.Kn)
				self.Bs = (self.Bs, self.Kn)
			elif isinstance(self.Is, tuple):
				self.Ks = tuple(list(self.Ks) + [self.Kn])
				self.Bs = tuple(list(self.Bs) + [self.Kn])
		self.K = PyNN.ParamInit(PyNN, self.Ks, alg, mean, sd, a, b)
		self.B = PyNN.ParamInit(PyNN, self.Bs, alg, mean, sd, a, b)
	def forward(self, x):
		self.x = x


		self.x = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[0,1,0],[1,0,1],[0,1,0]],[[2,1,2],[1,2,1],[2,1,2]]])
		self.K = np.array([[[[1, 0],[0, 1]],[[1, 1],[1, 1]],[[0, 1],[1, 0]]],[[[0, 1],[1, 0]],[[1, 0],[0, 1]],[[1, 1],[1, 1]]]])
		self.K = np.reshape(self.K, (2, 2, 3, 2))
		self.B = np.array([[[1, 2],[3, 4]],[[-1, -2],[-3, -4]]])



		if self.padding == 'same':
			fS = self.Ss
			self.x = PyNN.Padding(PyNN, self.x,
			kernel=self.Ks, stride=fS, val='zeros', alg='same')
		if isinstance(self.Is, int):
			C = self.Ks[0] if isinstance(self.Ks, tuple) else self.Ks
			windows = np.lib.stride_tricks.sliding_window_view(self.x, C)
			windows = windows[::self.Ss]
			self.y = np.dot(windows, self.K) + self.B
		if isinstance(self.Is, tuple):
			C = self.Ks[:-1]
			windows = np.lib.stride_tricks.sliding_window_view(self.x, C)
			slices = tuple(slice(None, None, s) for s in self.Ss)
			windows = windows[slices]
			dims = len(self.Ks[:-1] if self.Kn > 1 else self.Ks)
			if dims == 2:
				self.y = np.tensordot(self.K, windows, axes=dims) + self.B
			if dims == 3:
				self.y = np.tensordot(windows, self.K, axes=3) #+ self.B

		print(self.x.shape)
		print(self.K.shape)
		print(self.B.shape)


		return(self.y)



	def backward(self, dz):
		self.dK = np.zeros(self.Ks) # self.dw
		self.dB = np.copy(dz)       # self.db
		self.dx = np.zeros(self.x.shape)
		print(self.dK)
		print(self.dB)
		return(self.dx)

#x, dz = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([1, 3, 2, 5, 8, 7, 3]) ; C = Conv(input_shape=9, kernel_shape=3, kernel_number=2, stride_shape=1, padding='valid', alg='integers', a=0, b=9)
#x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) ; C = Conv(padding='valid', kernel_number=2, alg='integers', a=0, b=9 )
x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]]]) ; C = Conv(input_shape=(3,3,3), kernel_shape=(2,2,3), kernel_number=2, stride_shape=(1,1,1), padding='valid', alg='integers', a=0, b=9)
y = C.forward(x)
#dx = C.backward(dz)
print(y, y.shape)
#print(dx)
