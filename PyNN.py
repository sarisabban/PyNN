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
	def shuffle_data(self, X, Y):
		''' Shuffle X and Y in unison '''
		indices = np.random.permutation(X.shape[0])
		return(X[indices], Y[indices])
	def cost(self, loss):
		''' The cost function '''
		return(np.mean(loss))
	def forward(self, X, Y):
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
	class EarlyStopping():
		''' Early stopping function tracking training loss stagnation '''
		def __init__(self, min_delta=1e-4):
			self.min_delta = min_delta
			self.best_loss = float('inf')
		def check(self, epoch, loss):
			if loss < self.best_loss - self.min_delta: self.best_loss = loss
			if epoch > 1 and abs(loss - self.best_loss) < self.min_delta:
				print('Stopping early: Training loss plateaued')
				return(True)
			return(False)
	def show(self):
		''' Print out the structure of the network '''
		print('-'*60 + f"\n{'Layer':<25}{'Shape':<25}Parameters\n" + '-'*60)
		total_params = 0
		for layer in self.layers:
			name = layer.__class__.__name__
			if hasattr(layer, 'w'):
				shape = layer.w.shape
				params = math.prod(shape)
				total_params += params
			else:
				shape, params = '', ''
			print(f'{name:<25}{str(shape):<25}{params}')
		print('-'*30 + f'\nTotal Parameters: {total_params:,}')



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
#	run training loop without backprop
#	def GPU(self):
#	''' Use GPU instead of CPU '''





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
	#---------- Loss Functions ---------- #
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
	#---------- Accuracy Functions ---------- #
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
	#---------- Optimisers ---------- #
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



#	class Dropout():
#		''' The dropout regularisation layer '''
#		def __init__(self, p=0.25):
#			self.p = p
#		def forward(self, y, train=True):
#			if train:
#				self.mask = np.random.binomial(1, 1-self.p, y.shape)/(1-self.p)
#				y *= self.mask
#			return(y)
#		def backward(self, DA_Dz):
#			DA_Dz *= self.mask
#			return(DA_Dz)
#	class BatchNorm():
#		''' The Batch Normalisation regularisation layer '''
#		def __init__(self, g=1.0, b=0.0, e=1e-7): ##############3 not confident of it
#			self.w = g # replace g with w for optimiser
#			self.b = b
#			self.e = e
#		def forward(self, z):
#			self.z = z
#			self.mean = np.mean(z, axis=0, keepdims=True)
#			self.var = np.var(z, axis=0, keepdims=True)
#			self.z_norm = (self.z - self.mean) / np.sqrt(self.var + self.e)
#			z_new = self.w * self.z_norm + self.b
#			return(z_new)
#		def backward(self, DA_Dz):
#			m = self.z.shape[0]
#			self.dL_dw = np.sum(DA_Dz * self.z_norm, axis=0, keepdims=True) # dg
#			self.dL_db = np.sum(DA_Dz, axis=0, keepdims=True) # db
#			self.dB_dz = (self.w * (1./np.sqrt(self.var + self.e)) / m) * (m * DA_Dz - np.sum(DA_Dz, axis=0)
#			- (1./np.sqrt(self.var + self.e))**2 * (self.z - self.mean) * np.sum(DA_Dz*(self.z - self.mean), axis=0))
#			return(self.dB_dz)
#	class L1L2():
#		''' L1 + L2 regularisation '''
#		def forward(self, w, b, l1w, l1b, l2w, l2b):
#			self.w = w
#			self.b = b
#			self.L1w = l1w * np.sum(np.abs(w))
#			self.L1b = l1b * np.sum(np.abs(b))
#			self.L2w = l2w * np.sum(w**2)
#			self.L2b = l2b * np.sum(b**2)
#			L1L2 = self.L1w + self.L1b + self.L2w + self.L2b
#			return(L1L2)
#		def backward(self, dL_dw, dL_db):
#			dL1_dw = np.ones_like(w)
#			dL1_dw[w < 0] = -1
#			dL_dw += self.l1w * dL1_dw
#			dL1_db = np.ones_like(b)
#			dL1_db[b < 0] = -1
#			dL_db += self.l1b * dL1_db
#			dL_dw += 2 * self.l2w * w
#			dL_db += 2 * self.l2b * b
#			return(dL_dw, dL_db)





	#---------- Layers ----------#
	class Dense():
		''' A dense layer '''
		def __init__(self, inputs=1, outputs=1,
					alg='he uniform', mean=0.0, sd=0.1, a=-0.5, b=0.5):
			''' Initialise parameters '''
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
			z = np.dot(self.x, self.w) + self.b
			return(z)
		def backward(self, dz):
			self.dw = np.dot(self.x.T, dz)
			self.db = np.sum(dz, axis=0, keepdims=True)
			self.dx = np.dot(dz, self.w.T)
			return(self.dx)
	#---------- Training ---------- #
	def train(self,
			X_train=None, Y_train=None,
			X_valid=None, Y_valid=None,
			X_tests=None, Y_tests=None,
			batch_size = None,
			loss='BCE',
			accuracy='BINARY',
			optimiser='SGD', lr=0.1, decay=0.0, beta1=0.9, beta2=0.999, e=1e-7,
			early_stop=False,
			epochs=1):
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
			# Shuffle training datatset
			X_train, Y_train = self.shuffle_data(X_train, Y_train)
			for step in range(steps + 1):
				# Batch segmentation
				X_train_batch = X_train
				Y_train_batch = Y_train
				if batch_size is not None:
					X_train_batch = X[step * batch_size:(step + 1) * batch_size]
					Y_train_batch = Y[step * batch_size:(step + 1) * batch_size]
				# Forward propagation
				y_true = Y_train_batch
				y_pred = self.forward(X_train_batch, Y_train_batch)
				cost_train = self.cost(loss_fn.forward(y_true, y_pred))
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
			if early_stop and STOP.check(epoch, cost_train): break
			# Evaluate validation set
			if X_valid is not None and Y_valid is not None:
				y_true = Y_valid
				y_pred = self.forward(X_valid, Y_valid)
				cost_valid = self.cost(loss_fn.forward(y_true, y_pred))
				accuracy_valid = acc.calc(y_true, y_pred)
		# Evaluate test set
		if X_tests is not None and Y_tests is not None:
			y_true = Y_tests
			y_pred = self.forward(X_tests, Y_tests)
			cost_tests = self.cost(loss_fn.forward(y_true, y_pred))
			accuracy_tests = acc.calc(y_true, y_pred)


'''
[ ] Other utilities
[ ] general multi dimension tensor shapes
[ ] Verbosity
[ ] Regularisation
'''

#----- Import Data -----#
def sine_data(samples=1000):
	X = np.arange(samples).reshape(-1, 1) / samples
	y = np.sin(2 * np.pi * X).reshape(-1, 1)
	return(X, y)
X, Y = sine_data()

model = PyNN()
model.add(model.Dense(1, 64))
model.add(model.Sigmoid())
model.add(model.Dense(64, 64))
model.add(model.Sigmoid())
model.add(model.Dense(64, 1))
model.add(model.Linear())

model.show()
#model.train(X, Y, loss='MSE', accuracy='regression', lr=0.05, epochs=10)
