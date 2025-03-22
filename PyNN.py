import math
import numpy as np
np.random.seed(42)




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




class PyNN():
	''' Lightweight NumPy-based neural network library '''
	def __init__(self):
		''' Initialise the class with the following '''
		self.layers = []
	def add(self, layer):
		''' Add an object to the network '''
		self.layers.append(layer)
#	def set(self, *, loss, optimiser):
#		''' Set the loss and the optimiser '''
#		self.loss = loss
#		self.optimiser = optimiser
	def train(self, X, Y, batch_size=None, epochs=1, verbose=0):
		''' Train the network, forward pass followed by backward pass '''
		for epoch in range(epochs):
		# Mini-Baches
		# steps in mini batches

#			for layer in self.layers:
#				layer.forward(X)
			output = self.forward(X)

			print(output)



	def finalise(self):
		self.input_layer = Layer_Input()
		layer_count = len(self.layers)


		
		for i in range(layer_count):
			# If it's the first layer,
			# the previous layer object is the input layer
			if i == 0:
				self.layers[i].prev = self.input_layer
				self.layers[i].next = self.layers[i+1]
			# All layers except for the first and the last
			elif i < layer_count - 1:
				self.layers[i].prev = self.layers[i-1]
				self.layers[i].next = self.layers[i+1]
			# The last layer - the next object is the loss
			else:
				self.layers[i].prev = self.layers[i-1]
#				self.layers[i].next = self.loss


	def forward(self, X):
		# Call forward method on the input layer
		# this will set the output property that
		# the first layer in "prev" object is expecting
		self.input_layer.forward(X)
		# Call forward method of every object in a chain
		# Pass output of the previous object as a parameter
		for layer in self.layers:
			layer.forward(layer.prev.output)
		# "layer" is now the last object from the list,
		# return its output
		return layer.output



#	def show(self):
#		''' Print out the structure of the network '''
#	def verbosity(self):
#		''' Control level of information printout during training '''
#	def load(self, path):
#		''' Load model '''
#	def save(self, path):
#		''' Save model '''
#		with open(path, 'wb') as f:
#		pickle.dump(self.w, f)
#	def batch(self):
#		''' Setup mini-batches '''
#		steps = X.shape[0] // batch_size
#	def evaluate(self):
#	''' Train on training set, then validate on valid, after training test of test set '''
#	def checkpoint(self):
#	''' Add a training checkpoint '''
#	def predict(self):
#	''' Perform a prediction '''
#	def GPU(self):
#	''' Use GPU instead of CPU '''

# test in MNIST and fasion-MNIST





class Layer_Input():
	def forward(self, inputs):
		self.output = inputs


class Dense():
	''' A dense layer '''
	def __init__(self, inputs=1, outputs=1):
		''' Initialise parameters '''
		self.w, self.b, self.gamma, self.beta = Parameters(inputs, outputs)
	def forward(self, x):
		''' The forward pass '''
		self.z = np.dot(x, self.w) + self.b
#	def backward(self, y, DA_DZ, w):
#		''' The backward pass '''
#		dL_dw = np.dot(y.T, DA_Dz)
#		dL_db = np.sum(DA_Dz, axis=0, keepdims=True)
#		dL_dx = np.dot(DA_Dz, w.T)
#		return(dL_dw, dL_db, dL_dx)

class ReLU():
	''' The ReLU activation function '''
	def forward(self, z):
		self.y = np.maximum(0, z)
#	def backward(self):

def d_ReLU(DL_DY, z):# The derivative of the ReLU activation function
	DR_Dz = DL_DY.copy()
	DR_Dz[z <= 0] = 0
	return(DR_Dz)

class Linear():
	''' The Linear activation function '''
	def forward(self, z):
		self.y = z
#	def backward(self):

def d_Linear(DL_DY):# The derivative of the Linear activation function
	DA_Dz = DL_DY.copy()
	return(DA_Dz)

class Softmax():
	''' The Softmax activation function '''
	def forward(self, z):
		exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
		self.y = exp_values / np.sum(exp_values, axis=1, keepdims=True)
#	def backward(self):

def d_Softmax(DL_DY, y_pred):# Derivative of the Softmax activation function
	DS_Dz = np.empty_like(DL_DY)
	for i, (y, dy) in enumerate(zip(y_pred, DL_DY)):
		y = y.reshape(-1, 1)
		jacobian_matrix = np.diagflat(y) - np.dot(y, y.T)
		DS_Dz[i] = np.dot(jacobian_matrix, dy)
	return(DS_Dz)

class Sigmoid():
	''' The Sigmoid activation function '''
	def forward(self, z):
		self.y = 1 / (1 + np.exp(-z))
#	def backward(self):

def d_Sigmoid(DA_Dz, y_pred):# The derivative of the Sigmoid activation function
	DS_Dz = DA_Dz * (1 - y_pred) * y_pred
	return(DS_Dz)

#class TanH():
#	''' The TanH activation function '''
#	def forward(self, z):
#	def backward(self):
#class Step():
#	''' The Step activation function '''
#	def forward(self, z):
#	def backward(self):



class BCE_Loss():
	''' The Binary Cross-Entropy loss function '''
	def forward(y_true, y_pred):
		y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
		loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
		return(loss)

#def d_BCE_Loss(y_true,  y_pred):# Derivative of Binary Cross-Entropy loss
#	y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
#	DL_DY = (y_pred - y_true) / (y_pred * (1-y_pred)) / len(y_pred)
#	return(DL_DY)




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



model = PyNN()
model.add(Dense(2, 64))
model.add(ReLU())
model.add(Dense(64, 1))
model.add(Linear())

Y = Y.reshape(-1, 1) # y = (200, 1)

model.finalise()


model.train(X, Y)

#print(model.layers[0].w.shape)












def CCE_Loss(y_true, y_pred):# The Categorical Cross-Entropy loss
	y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
	y_pred_vector = y_pred[range(len(y_pred)), y_true]
	loss = -np.log(y_pred_vector)
	return(loss)

def d_CCE_Loss(y_true, y_pred):# Derivative of Categorical Cross-Entropy loss
	if len(y_true.shape) == 1:
		y_true = np.eye(len(y_pred[0]))[y_true]
	DL_DY = (-y_true / y_pred) / len(y_pred)
	return(DL_DY)




def Cost(loss):# The Cost function
	cost = np.mean(loss)
	return(cost)

def Categorical_Accuracy(y_true, y_pred):# Measure the accuracy
	predictions = np.argmax(y_pred, axis=1)
	accuracy = np.mean(predictions==y_true)
	return(accuracy)

def Binary_Accuracy(y_true, y_pred):# Measure the accuracy
	predictions = (y_pred > 0.5) * 1
	accuracy = np.mean(predictions==y_true)
	return(accuracy)

def SGD(lr, w, b, dL_dw, dL_db):# Stochastic Gradient Descent
	w -= lr * dL_dw
	b -= lr * dL_db
	return(w, b)

def Adam(lr,decay,beta1,beta2,e,w,b,dL_dw,dL_db,iters,w_m,w_c,b_m,b_c):# Adam Gradient Descent
	lr = lr * (1. / (1. + decay * iters))
	w_m = beta1 * w_m + (1 - beta1) * dL_dw
	b_m = beta1 * b_m + (1 - beta1) * dL_db
	w_m_c = w_m / (1 - beta1 ** (iters + 1))
	b_m_c = b_m / (1 - beta1 ** (iters + 1))
	w_c = beta2 * w_c + (1 - beta2) * dL_dw**2
	b_c = beta2 * b_c + (1 - beta2) * dL_db**2
	w_c_c = w_c / (1 - beta2 ** (iters + 1))
	b_c_c = b_c / (1 - beta2 ** (iters + 1))
	w -= lr * w_m_c / (np.sqrt(w_c_c) + e)
	b -= lr * b_m_c / (np.sqrt(b_c_c) + e)
	return(w, b, w_m, w_c, b_m, b_c)

def L1L2(w, b, l1w, l1b, l2w, l2b):# L1 + L2 regularisation
	L1w = l1w * np.sum(np.abs(w))
	L1b = l1b * np.sum(np.abs(b))
	L2w = l2w * np.sum(w**2)
	L2b = l2b * np.sum(b**2)
	L1L2 = L1w + L1b + L2w + L2b
	return(L1L2)

def d_L1L2(w, b, dL_dw, dL_db, l1w, l1b, l2w, l2b):# Derivative of L1 + L2 regularisation
	#L1 on weights
	dL1_dw = np.ones_like(w)
	dL1_dw[w < 0] = -1
	dL_dw += l1w * dL1_dw
	#L1 on biases
	dL1_db = np.ones_like(b)
	dL1_db[b < 0] = -1
	dL_db += l1b * dL1_db
	#L2 on weights
	dL_dw += 2 * l2w * w
	#L2 on biases
	dL_db += 2 * l2b * b
	return(dL_dw, dL_db)

def Dropout(p, y):# Dropout layer - prediction
	y *= np.random.binomial(1, 1-p, y.shape) / (1-p)
	return(y)

def d_Dropout(p, DA_Dz):# Dropout layer - derivative
	DA_Dz *= np.random.binomial(1, 1-p, DA_Dz.shape) / (1-p)
	return(DA_Dz)


def MSE_Loss(y_true, y_pred):# The Mean Squared Error loss
	loss = (y_true - y_pred)**2
	return(loss)

def d_MSE_Loss(y_true, y_pred):# The derivative of the Mean Squared Error loss
	DL_DY = (-2 * (y_true - y_pred) / len(y_pred[0])) / len(y_pred)
	return(DL_DY)

def MAE_Loss(y_true, y_pred):# The Mean Absolute Error loss
	loss = np.abs(y_true - y_pred)
	return(loss)

def d_MAE_Loss(y_true, y_pred):# The derivative of the Mean Absolute Error loss
	DL_DY = (np.sign(y_true - y_pred) / len(y_pred[0])) / len(y_pred)
	return(DL_DY)

def Regression_Accuracy(y_true, y_pred):# Measure the accuracy
	accuracy_precision = np.std(y_true) / 250
	predictions = y_pred
	accuracy = np.mean(np.absolute(predictions - y_true) < accuracy_precision)
	return(accuracy)

def BatchNorm(z, g, b, e=1e-7):
	mean = np.mean(z, axis=0, keepdims=True)
	var = np.var(z, axis=0, keepdims=True)
	z_norm = g * (z - mean) / np.sqrt(var + e) + b
	cache = (z, mean, var, g, b, e)
	return(z_norm, cache)

def d_BatchNorm(DA_Dz, z_norm, cache):
	z, mean, var, g, b, e = cache
	m = z.shape[0]
	dg = np.sum(DA_Dz * z_norm, axis=0, keepdims=True)
	db = np.sum(DA_Dz, axis=0, keepdims=True)
	dB_dz = (g * (1./np.sqrt(var + e)) / m) * (m * DA_Dz - np.sum(DA_Dz, axis=0)
	- (1./np.sqrt(var + e))**2 * (z - mean) * np.sum(DA_Dz*(z - mean), axis=0))
	return(dB_dz, dg, db)






