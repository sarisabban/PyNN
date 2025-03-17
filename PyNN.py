import math
import numpy as np
np.random.seed(42)

#----- Import Data -----#
def spiral_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    Y = np.zeros(samples*classes, dtype='uint8')
    for class_n in range(classes):
        ix = range(samples*class_n, samples*(class_n+1))
        r=np.linspace(0.0, 1, samples)
        t=np.linspace(class_n*4,(class_n+1)*4,samples)+np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        Y[ix] = class_n
    return(X, Y)

def sine_data(samples=1000):
    X = np.arange(samples).reshape(-1, 1) / samples
    Y = np.sin(2 * np.pi * X).reshape(-1, 1)
    return(X, Y)

def Dense(n_inputs, n_outputs, x, w, b):# A single Dense layer
	z = np.dot(x, w) + b
	return(z, w, b)

def d_Dense(y, DA_Dz, w):# The derivative of a single Dense layer
	dL_dw = np.dot(y.T, DA_Dz)
	dL_db = np.sum(DA_Dz, axis=0, keepdims=True)
	dL_dx = np.dot(DA_Dz, w.T)
	return(dL_dw, dL_db, dL_dx)

def ReLU(z):# The ReLU activation function
	y = np.maximum(0, z)
	return(y)

def d_ReLU(DL_DY, z):# The derivative of the ReLU activation function
	DR_Dz = DL_DY.copy()
	DR_Dz[z <= 0] = 0
	return(DR_Dz)

def Softmax(x):# The Softmax activation function
	exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
	y = exp_values / np.sum(exp_values, axis=1, keepdims=True)
	return(y)

def d_Softmax(DL_DY, y_pred):# Derivative of the Softmax activation function
	DS_Dz = np.empty_like(DL_DY)
	for i, (y, dy) in enumerate(zip(y_pred, DL_DY)):
		y = y.reshape(-1, 1)
		jacobian_matrix = np.diagflat(y) - np.dot(y, y.T)
		DS_Dz[i] = np.dot(jacobian_matrix, dy)
	return(DS_Dz)

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

def Sigmoid(z):# The Sigmoid activation function
	y = 1 / (1 + np.exp(-z))
	return(y)

def d_Sigmoid(DA_Dz, y_pred):# The derivative of the Sigmoid activation function
	DS_Dz = DA_Dz * (1 - y_pred) * y_pred
	return(DS_Dz)

def BCE_Loss(y_true, y_pred):# The Binary Cross Entropy loss
	y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
	loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
	return(loss)

def d_BCE_Loss(y_true,  y_pred):# Derivative of Binary Cross-Entropy loss
	y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
	DL_DY = (y_pred - y_true) / (y_pred * (1-y_pred)) / len(y_pred)
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

def Linear(z):# The Linear activation function
	y = z
	return(y)

def d_Linear(DL_DY):# The derivative of the Linear activation function
	DA_Dz = DL_DY.copy()
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

def Parameters(inputs=3, outputs=3, alg='he uniform', sd=0.1, a=-0.5, b=0.5):
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



# Multi class
#X, Y = spiral_data(samples=100, classes=3) # X=(300, 2)   Y=(300,)   3 Classes
# Binary class
#X, Y = spiral_data(samples=100, classes=2) # X = (200, 2)   y = (200,)
# Regression
#X, Y = sine_data() # X = (1000, 1)   Y = (1000, 1)



#----- Training -----#
"""
# Multi-Classification
w1 = 0.01 * np.random.randn(2, 64)
b1 = np.zeros((1, 64))
w2 = 0.01 * np.random.randn(64, 3)
b2 = np.zeros((1, 3))

w1_m = np.zeros_like(w1)
w1_c = np.zeros_like(w1)
b1_m = np.zeros_like(b1)
b1_c = np.zeros_like(b1)
w2_m = np.zeros_like(w2)
w2_c = np.zeros_like(w2)
b2_m = np.zeros_like(b2)
b2_c = np.zeros_like(b2)

for epoch in range(1000):
	y_true = Y
	z1, w1, b1 = Dense(2, 64, X, w1, b1)
	y1 = ReLU(z1)
	y1 = Dropout(0.1, y1) # dropout
	z2, w2, b2 = Dense(64, 3, y1, w2, b2)
	y_pred = Softmax(z2)
	reg1 = L1L2(w1, b1, 0, 0, 5e-4, 5e-4) # L1 + L2
	reg2 = L1L2(w2, b2, 0, 0, 0, 0) # L1 + L2
	loss = CCE_Loss(y_true, y_pred)
	cost, accuracy = Cost(loss) + reg1 + reg2, Accuracy(y_true, y_pred)
	cost, accuracy = Cost(loss), Categorical_Accuracy(y_true, y_pred)
	DL_DY = d_CCE_Loss(y_true, y_pred)
	DS_Dz2 = d_Softmax(DL_DY, y_pred)
	dL_dwL2, dL_dbL2, DS_Dy1 = d_Dense(y1, DS_Dz2, w2)
	DS_Dy1 = d_Dropout(0.1, DS_Dy1)
	dL_dwL2, dL_dbL2 = d_L1L2(w2, b2, dL_dwL2, dL_dbL2, 0, 0, 0, 0) # L1 + L2
	DR_Dz1 = d_ReLU(DS_Dy1, z1)
	dL_dwL1, dL_dbL1, dL_dx = d_Dense(X, DR_Dz1, w1)
	dL_dwL1, dL_dbL1 = d_L1L2(w1, b1, dL_dwL1, dL_dbL1, 0, 0, 5e-4, 5e-4) # L1 + L2
	w1, b1, w1_m, w1_c, b1_m, b1_c = Adam(0.05, 5e-7, 0.9, 0.999, 1e-7, w1, b1, dL_dwL1, dL_dbL1, epoch+1, w1_m, w1_c, b1_m, b1_c)
	w2, b2, w2_m, w2_c, b2_m, b2_c = Adam(0.05, 5e-7, 0.9, 0.999, 1e-7, w2, b2, dL_dwL2, dL_dbL2, epoch+1, w2_m, w2_c, b2_m, b2_c)
	print(f'epoch: {epoch+1} --- cost: {cost:.7f} --- accuracy: {accuracy:.3f}')
"""


"""
# Binary-Classification
w1 = 0.01 * np.random.randn(2, 64)
b1 = np.zeros((1, 64))
w2 = 0.01 * np.random.randn(64, 1)
b2 = np.zeros((1, 1))

w1_m = np.zeros_like(w1)
w1_c = np.zeros_like(w1)
b1_m = np.zeros_like(b1)
b1_c = np.zeros_like(b1)
w2_m = np.zeros_like(w2)
w2_c = np.zeros_like(w2)
b2_m = np.zeros_like(b2)
b2_c = np.zeros_like(b2)

Y = Y.reshape(-1, 1) # y = (200, 1)
for epoch in range(1):
	y_true = Y
	z1, w1, b1 = Dense(2, 64, X, w1, b1)
	y1 = ReLU(z1)
	z2, w2, b2 = Dense(64, 1, y1, w2, b2)
	y_pred = Sigmoid(z2)
	reg1 = L1L2(w1, b1, 0, 0, 5e-4, 5e-4) # L1 + L2
	reg2 = L1L2(w2, b2, 0, 0, 0, 0) # L1 + L2
	loss = BCE_Loss(y_true, y_pred)
	cost, accuracy = Cost(loss) + reg1 + reg2, Binary_Accuracy(y_true, y_pred)
	DL_DY = d_BCE_Loss(y_true, y_pred)
	DS_Dz2 = d_Sigmoid(DL_DY, y_pred)
	dL_dwL2, dL_dbL2, DS_Dy1 = d_Dense(y1, DS_Dz2, w2)
	dL_dwL2, dL_dbL2 = d_L1L2(w2, b2, dL_dwL2, dL_dbL2, 0, 0, 0, 0) # L1 + L2
	DR_Dz1 = d_ReLU(DS_Dy1, z1)
	dL_dwL1, dL_dbL1, dL_dx = d_Dense(X, DR_Dz1, w1)
	dL_dwL1, dL_dbL1 = d_L1L2(w1, b1, dL_dwL1, dL_dbL1, 0, 0, 5e-4, 5e-4) # L1 + L2
	w1, b1, w1_m, w1_c, b1_m, b1_c = Adam(0.05, 5e-7, 0.9, 0.999, 1e-7, w1, b1, dL_dwL1, dL_dbL1, epoch+1, w1_m, w1_c, b1_m, b1_c)
	w2, b2, w2_m, w2_c, b2_m, b2_c = Adam(0.05, 5e-7, 0.9, 0.999, 1e-7, w2, b2, dL_dwL2, dL_dbL2, epoch+1, w2_m, w2_c, b2_m, b2_c)
	print(f'epoch: {epoch+1} --- cost: {cost:.3f} --- accuracy: {accuracy:.3f}')
"""


"""
# Regression
w1 = 0.01 * np.random.randn(1, 64)
b1 = np.zeros((1, 64))
w2 = 0.01 * np.random.randn(64, 64)
b2 = np.zeros((1, 64))
w3 = 0.01 * np.random.randn(64, 1)
b3 = np.zeros((1, 1))

w1_m = np.zeros_like(w1)
w1_c = np.zeros_like(w1)
b1_m = np.zeros_like(b1)
b1_c = np.zeros_like(b1)
w2_m = np.zeros_like(w2)
w2_c = np.zeros_like(w2)
b2_m = np.zeros_like(b2)
b2_c = np.zeros_like(b2)
w3_m = np.zeros_like(w3)
w3_c = np.zeros_like(w3)
b3_m = np.zeros_like(b3)
b3_c = np.zeros_like(b3)

for epoch in range(5000):
	y_true = Y
	z1, w1, b1 = Dense(2, 64, X, w1, b1)
	y1 = ReLU(z1)
	z2, w2, b2 = Dense(64, 64, y1, w2, b2)
	y2 = ReLU(z2)
	z3, w3, b3 = Dense(64, 1, y2, w3, b3)
	y_pred = Linear(z3)
	loss = MSE_Loss(y_true, y_pred)
	cost, accuracy = Cost(loss), Regression_Accuracy(y_true, y_pred)
	DL_DY = d_MSE_Loss(y_true, y_pred)
	DS_Dz3 = d_Linear(DL_DY)
	dL_dwL3, dL_dbL3, DS_Dy2 = d_Dense(y2, DS_Dz3, w3)
	DR_Dz2 = d_ReLU(DS_Dy2, z2)
	dL_dwL2, dL_dbL2, DS_Dy1 = d_Dense(y1, DR_Dz2, w2)
	DR_Dz1 = d_ReLU(DS_Dy1, z1)
	dL_dwL1, dL_dbL1, dL_dx = d_Dense(X, DR_Dz1, w1)
	w1, b1, w1_m, w1_c, b1_m, b1_c = Adam(0.005, 1e-3, 0.9, 0.999, 1e-7, w1, b1, dL_dwL1, dL_dbL1, epoch+1, w1_m, w1_c, b1_m, b1_c)
	w2, b2, w2_m, w2_c, b2_m, b2_c = Adam(0.005, 1e-3, 0.9, 0.999, 1e-7, w2, b2, dL_dwL2, dL_dbL2, epoch+1, w2_m, w2_c, b2_m, b2_c)
	w3, b3, w3_m, w3_c, b3_m, b3_c = Adam(0.005, 1e-3, 0.9, 0.999, 1e-7, w3, b3, dL_dwL3, dL_dbL3, epoch+1, w3_m, w3_c, b3_m, b3_c)
	print(f'epoch: {epoch+1} --- cost: {cost:.3f} --- accuracy: {accuracy:.3f}')
"""
