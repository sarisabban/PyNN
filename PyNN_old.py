

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
#	run training loop without backprop
#	def GPU(self):
#	''' Use GPU instead of CPU '''


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
		def backward(self, DA_Dz):
			DA_Dz *= self.mask
			return(DA_Dz)
	class BatchNorm():
		''' The Batch Normalisation regularisation layer '''
		def __init__(self, g=1.0, b=0.0, e=1e-7): ##############3 not confident of it
			self.w = g # replace g with w for optimiser
			self.b = b
			self.e = e
		def forward(self, z):
			self.z = z
			self.mean = np.mean(z, axis=0, keepdims=True)
			self.var = np.var(z, axis=0, keepdims=True)
			self.z_norm = (self.z - self.mean) / np.sqrt(self.var + self.e)
			z_new = self.w * self.z_norm + self.b
			return(z_new)
		def backward(self, DA_Dz):
			m = self.z.shape[0]
			self.dL_dw = np.sum(DA_Dz * self.z_norm, axis=0, keepdims=True) # dg
			self.dL_db = np.sum(DA_Dz, axis=0, keepdims=True) # db
			self.dB_dz = (self.w * (1./np.sqrt(self.var + self.e)) / m) * (m * DA_Dz - np.sum(DA_Dz, axis=0)
			- (1./np.sqrt(self.var + self.e))**2 * (self.z - self.mean) * np.sum(DA_Dz*(self.z - self.mean), axis=0))
			return(self.dB_dz)
	class L1L2():
		''' L1 + L2 regularisation '''
		def forward(self, w, b, l1w, l1b, l2w, l2b):
			self.w = w
			self.b = b
			self.L1w = l1w * np.sum(np.abs(w))
			self.L1b = l1b * np.sum(np.abs(b))
			self.L2w = l2w * np.sum(w**2)
			self.L2b = l2b * np.sum(b**2)
			L1L2 = self.L1w + self.L1b + self.L2w + self.L2b
			return(L1L2)
		def backward(self, dL_dw, dL_db):
			dL1_dw = np.ones_like(w)
			dL1_dw[w < 0] = -1
			dL_dw += self.l1w * dL1_dw
			dL1_db = np.ones_like(b)
			dL1_db[b < 0] = -1
			dL_db += self.l1b * dL1_db
			dL_dw += 2 * self.l2w * w
			dL_db += 2 * self.l2b * b
			return(dL_dw, dL_db)
