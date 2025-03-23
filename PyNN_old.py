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
