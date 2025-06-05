#sys and core
import sys
sys.path.insert(0, "../")
from core import *
#config
import config as c


class convexLinear(torch.nn.Module):
	"""

	Custom linear layer with positive weights and no bias

	Init:	size_in, size_out
	Inputs: input data
	Output: input data times softplus(trainable weight)

	"""
	def __init__(self, size_in, size_out, ):
		super().__init__()
		self.size_in, self.size_out = size_in, size_out
		weights = torch.Tensor(size_out, size_in)
		self.weights = torch.nn.Parameter(weights)

		# initialize weights
		torch.nn.init.kaiming_uniform_(self.weights, a=np.emath.sqrt(5))

	def forward(self, x):
		w_times_x= torch.mm(x, torch.nn.functional.softplus(self.weights.t()))
		return w_times_x

class ICNN(torch.nn.Module):
	"""

	Material model based on Input convex neural network.

	Initialize:
	n_input:			Input layer size
	n_hidden:			Hidden layer size / number of neurons
	n_output:			Output layer size
	use_dropout:		Activate dropout during training
	dropout_rate:		Dropout probability.
	anisotropy_flag:	{single, double} -> type of fiber families
	fiber_type:			{mirror, general} -> type of fiber arrangement in case of two (or more) fiber families.

	Inputs: 			Deformation gradient in the form: (F11,F12,F21,F22)
	Output: 			NN-based strain energy density (W_NN)

	"""
	def __init__(self, n_input, n_hidden, n_output, use_dropout, dropout_rate, anisotropy_flag=None, fiber_type=None):
		super(ICNN, self).__init__()

		# Create Module dicts for the hidden and skip-connection layers
		self.layers = torch.nn.ModuleDict()
		self.skip_layers = torch.nn.ModuleDict()
		self.global_pooling= torch.nn.ModuleDict()
		self.depth = len(n_hidden)
		self.dropout = c.use_dropout
		self.p_dropout = c.dropout_rate
		self.anisotropy_flag = anisotropy_flag
		self.fiber_type = fiber_type

		if self.anisotropy_flag is not None:
			self.alpha = torch.nn.Parameter((torch.randn(1,1)))#torch.tensor([0.25])#

		self.layers[str(0)] = torch.nn.Linear(n_input, n_hidden[0]).float()
		# Create create NN with number of elements in n_hidden as depth
		for i in range(1, self.depth):
			self.layers[str(i)] = convexLinear(n_hidden[i-1], n_hidden[i]).float()
			self.skip_layers[str(i)] = torch.nn.Linear(n_input, n_hidden[i]).float()

		self.layers[str(self.depth)] = convexLinear(n_hidden[self.depth-1], n_output).float()
		self.skip_layers[str(self.depth)] = convexLinear(n_input, n_output).float()

		self.global_pooling =torch.nn.AdaptiveAvgPool1d(output_size=1)

	def forward(self, x):
		# Get angle
		if self.anisotropy_flag is not None:
			pi = 3.141592653589732
			if self.anisotropy_flag == 'single':
				alpha = pi*torch.sigmoid(self.alpha)
			elif self.anisotropy_flag == 'double':
				alpha = pi/2*torch.sigmoid(self.alpha)

		# Get F components
		F11 = x[:,0:1]
		F12 = x[:,1:2]
		F21 = x[:,2:3]
		F22 = x[:,3:4]

		# Compute right Cauchy green strain Tensor
		C11 = F11**2. + F21**2. 
		C12 = F11*F12 + F21*F22
		C21 = F11*F12 + F21*F22
		C22 = F12**2. + F22**2.

		# Compute computeStrainInvariants
		I1 = C11 + C22 + 1.0
		I2 = C11 + C22 - C12*C21 + C11*C22
		I3 = C11*C22 - C12*C21
		if self.anisotropy_flag is not None:
			Ia = torch.cos(alpha)*(C11*torch.cos(alpha)+C12*torch.sin(alpha)) + torch.sin(alpha)*(C21*torch.cos(alpha)+C22*torch.sin(alpha))

		if self.anisotropy_flag == 'double':
			if self.fiber_type == 'mirror':
				beta = -alpha
			elif self.fiber_type == 'general':
				beta = alpha + pi/2.
			Ib = torch.cos(beta)*(C11*torch.cos(beta)+C12*torch.sin(beta)) + torch.sin(beta)*(C21*torch.cos(beta)+C22*torch.sin(beta))

		# Apply transformation to invariants
		K1 = I1 * torch.pow(I3,-1./3.) - 3.0
		K2 = (I1 + I3 - 1.) * torch.pow(I3,-2./3.) - 3.0
		J = torch.sqrt(I3)
		K3 = (J-1.)**2

		if self.anisotropy_flag is not None:
			if self.anisotropy_flag == 'single':
				K4 = (Ia * J**(-2./3.) - 1.0)**2
				x_input = torch.cat((K1,K2,K3,K4),dim=1).float()

			elif self.anisotropy_flag == 'double':
				K4 = (Ia * J**(-2./3.) - 1.0)**2
				K5 = (Ib * J**(-2./3.) - 1.0)**2
				# Concatenate feature
				x_input = torch.cat((K1,K2,K3,K4,K5),dim=1).float()
		else:
			# Concatenate feature
			x_input = torch.cat((K1,K2,K3),dim=1).float()

		z = x_input.clone()
		z = self.layers[str(0)](z)

		for layer in range(1,self.depth):
			skip = self.skip_layers[str(layer)](x_input)
			z = self.layers[str(layer)](z)
			z += skip
			z = torch.nn.functional.softplus(z)
			if c.use_sftpSquared:
				z = c.scaling_sftpSq*torch.square(z)
			if self.training:
				if self.dropout:
					z = torch.nn.functional.dropout(z,p=self.p_dropout)
		y = self.layers[str(self.depth)](z) + self.skip_layers[str(self.depth)](x_input)

		z=self.global_pooling(y)
		print(z.shape)
		return y

def init_weights(m):
	if isinstance(m, torch.nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight)

def print_model_arch(model):
	print('\n\n','-'*80,'\n',model,'\n','-'*80,'\n\n')

def print_model_params(model):
	print('\n\n====================================================================')
	print('Initial model:')
	print(model.state_dict())
	print('====================================================================\n\n')




#.
class ICNN3D(torch.nn.Module):
	"""

	Material model based on Input convex neural network.

	Initialize:
	n_input:			Input layer size
	n_hidden:			Hidden layer size / number of neurons
	n_output:			Output layer size
	use_dropout:		Activate dropout during training
	dropout_rate:		Dropout probability.
	anisotropy_flag:	{single, double} -> type of fiber families
	fiber_type:			{mirror, general} -> type of fiber arrangement in case of two (or more) fiber families.

	Inputs: 			Deformation gradient in the form: (F11,F12,F21,F22)
	Output: 			NN-based strain energy density (W_NN)

	"""
	def __init__(self, n_input, n_hidden, n_output, use_dropout, dropout_rate, anisotropy_flag=None, fiber_type=None):
		super(ICNN3D, self).__init__()

		# Create Module dicts for the hidden and skip-connection layers
		self.layers = torch.nn.ModuleDict()
		self.skip_layers = torch.nn.ModuleDict()
		self.depth = len(n_hidden)
		self.dropout = c.use_dropout
		self.p_dropout = c.dropout_rate
		self.anisotropy_flag = anisotropy_flag
		self.fiber_type = fiber_type

		if self.anisotropy_flag is not None:
			self.alpha = torch.nn.Parameter((torch.randn(1,1)))#torch.tensor([0.25])#

		self.layers[str(0)] = torch.nn.Linear(n_input, n_hidden[0]).float()
		# Create create NN with number of elements in n_hidden as depth
		for i in range(1, self.depth):
			self.layers[str(i)] = convexLinear(n_hidden[i-1], n_hidden[i]).float()
			self.skip_layers[str(i)] = torch.nn.Linear(n_input, n_hidden[i]).float()

		self.layers[str(self.depth)] = convexLinear(n_hidden[self.depth-1], n_output).float()
		self.skip_layers[str(self.depth)] = convexLinear(n_input, n_output).float()



	def forward(self, x):
		# Get angle
		if self.anisotropy_flag is not None:
			pi = 3.141592653589732
			if self.anisotropy_flag == 'single':
				alpha = pi*torch.sigmoid(self.alpha)
			elif self.anisotropy_flag == 'double':
				alpha = pi/2*torch.sigmoid(self.alpha)

			# assume x has shape (batch_size, 9), columns = [F11, F12, F13, F21, F22, F23, F31, F32, F33]
		# For clarity, we slice out each component as a (batch_size×1) tensor:
		F11 = x[:, 0:1]  # shape = (B,1)
		F12 = x[:, 1:2]
		F13 = x[:, 2:3]

		F21 = x[:, 3:4]
		F22 = x[:, 4:5]
		F23 = x[:, 5:6]

		F31 = x[:, 6:7]
		F32 = x[:, 7:8]
		F33 = x[:, 8:9]

		# 1) Build the Right Cauchy‐Green tensor C = F^T F, component‐wise:
		#    C11 = F11^2 + F21^2 + F31^2
		C11 = F11**2 + F21**2 + F31**2

		#    C12 = F11·F12 + F21·F22 + F31·F32
		C12 = F11*F12 + F21*F22 + F31*F32

		#    C13 = F11·F13 + F21·F23 + F31·F33
		C13 = F11*F13 + F21*F23 + F31*F33

		#    C22 = F12^2 + F22^2 + F32^2
		C22 = F12**2 + F22**2 + F32**2

		#    C23 = F12·F13 + F22·F23 + F32·F33
		C23 = F12*F13 + F22*F23 + F32*F33

		#    C33 = F13^2 + F23^2 + F33^2
		C33 = F13**2 + F23**2 + F33**2

		# (Note: C21 = C12, C31 = C13, C32 = C23, but we only need them for invariants.)

		# 2) Compute the three invariants of C:

		#   I1 = trace(C) = C11 + C22 + C33
		I1 = C11 + C22 + C33

		#   I2 = sum of principal 2×2 minors of C:
		#        I2 =  C11·C22 + C11·C33 + C22·C33  –  (C12^2 + C13^2 + C23^2)
		I2 = (C11 * C22) + (C11 * C33) + (C22 * C33) - (C12**2 + C13**2 + C23**2)

		#   I3 = det(C).  But det(C) = (det F)².  So first form det(F):
		detF = (
			F11 * (F22*F33 - F23*F32)
		- F12 * (F21*F33 - F23*F31)
		+ F13 * (F21*F32 - F22*F31)
		)
		I3 = detF**2

		# 3) Now form the modified invariants K1, K2, K3:

		#    J  = sqrt(I3)
		J = torch.sqrt(I3)

		#    K1 = I1 * I3^(–1/3) – 3.0
		K1 = I1 * torch.pow(I3, -1.0/3.0) - 3.0

		#    K2 = I2 * I3^(–2/3) – 3.0
		K2 = I2 * torch.pow(I3, -2.0/3.0) - 3.0

		#    K3 = (J - 1.0)^2
		K3 = (J - 1.0)**2


		# Concatenate feature
		x_input = torch.cat((K1,K2,K3),dim=1).float()

		z = x_input.clone()
		z = self.layers[str(0)](z)

		for layer in range(1,self.depth):
			skip = self.skip_layers[str(layer)](x_input)
			z = self.layers[str(layer)](z)
			z += skip
			z = torch.nn.functional.softplus(z)
			if c.use_sftpSquared:
				z = c.scaling_sftpSq*torch.square(z)
			if self.training:
				if self.dropout:
					z = torch.nn.functional.dropout(z,p=self.p_dropout)
		y = self.layers[str(self.depth)](z) + self.skip_layers[str(self.depth)](x_input)
		return y


class ICNN3D_position(torch.nn.Module):
	"""

	Material model based on Input convex neural network.

	Initialize:
	n_input:			Input layer size
	n_hidden:			Hidden layer size / number of neurons
	n_output:			Output layer size
	use_dropout:		Activate dropout during training
	dropout_rate:		Dropout probability.
	anisotropy_flag:	{single, double} -> type of fiber families
	fiber_type:			{mirror, general} -> type of fiber arrangement in case of two (or more) fiber families.

	Inputs: 			Deformation gradient in the form: (F11,F12,F21,F22)
	Output: 			NN-based strain energy density (W_NN)

	"""
	def __init__(self, n_input, n_hidden, n_output, use_dropout, dropout_rate, anisotropy_flag=None, fiber_type=None):
		super(ICNN3D_position, self).__init__()

		# Create Module dicts for the hidden and skip-connection layers
		self.layers = torch.nn.ModuleDict()
		self.skip_layers = torch.nn.ModuleDict()
		self.depth = len(n_hidden)
		self.dropout = c.use_dropout
		self.p_dropout = c.dropout_rate
		self.anisotropy_flag = anisotropy_flag
		self.fiber_type = fiber_type

		if self.anisotropy_flag is not None:
			self.alpha = torch.nn.Parameter((torch.randn(1,1)))#torch.tensor([0.25])#

		self.layers[str(0)] = torch.nn.Linear(n_input, n_hidden[0]).float()
		# Create create NN with number of elements in n_hidden as depth
		for i in range(1, self.depth):
			self.layers[str(i)] = convexLinear(n_hidden[i-1], n_hidden[i]).float()
			self.skip_layers[str(i)] = torch.nn.Linear(n_input, n_hidden[i]).float()

		self.layers[str(self.depth)] = convexLinear(n_hidden[self.depth-1], n_output).float()
		self.skip_layers[str(self.depth)] = convexLinear(n_input, n_output).float()

	def forward(self, x):
		# Get angle
		if self.anisotropy_flag is not None:
			pi = 3.141592653589732
			if self.anisotropy_flag == 'single':
				alpha = pi*torch.sigmoid(self.alpha)
			elif self.anisotropy_flag == 'double':
				alpha = pi/2*torch.sigmoid(self.alpha)

		#Part of the input is the centroids, to avoid assuming homogenenous materials

		# For clarity, we slice out each component as a (batch_size×1) tensor:
		F11 = x[:, 0:1]  # shape = (B,1)
		F12 = x[:, 1:2]
		F13 = x[:, 2:3]

		F21 = x[:, 3:4]
		F22 = x[:, 4:5]
		F23 = x[:, 5:6]

		F31 = x[:, 6:7]
		F32 = x[:, 7:8]
		F33 = x[:, 8:9]

		#Centroid positions
		Xc  = x[:,  9:10]
		Yc  = x[:, 10:11]
		Zc  = x[:, 11:12]

		# 1) Build the Right Cauchy‐Green tensor C = F^T F, component‐wise:
		#    C11 = F11^2 + F21^2 + F31^2
		C11 = F11**2 + F21**2 + F31**2

		#    C12 = F11·F12 + F21·F22 + F31·F32
		C12 = F11*F12 + F21*F22 + F31*F32

		#    C13 = F11·F13 + F21·F23 + F31·F33
		C13 = F11*F13 + F21*F23 + F31*F33

		#    C22 = F12^2 + F22^2 + F32^2
		C22 = F12**2 + F22**2 + F32**2

		#    C23 = F12·F13 + F22·F23 + F32·F33
		C23 = F12*F13 + F22*F23 + F32*F33

		#    C33 = F13^2 + F23^2 + F33^2
		C33 = F13**2 + F23**2 + F33**2

		# (Note: C21 = C12, C31 = C13, C32 = C23, but we only need them for invariants.)

		# 2) Compute the three invariants of C:

		#   I1 = trace(C) = C11 + C22 + C33
		I1 = C11 + C22 + C33

		#   I2 = sum of principal 2×2 minors of C:
		#        I2 =  C11·C22 + C11·C33 + C22·C33  –  (C12^2 + C13^2 + C23^2)
		I2 = (C11 * C22) + (C11 * C33) + (C22 * C33) - (C12**2 + C13**2 + C23**2)

		#   I3 = det(C).  But det(C) = (det F)².  So first form det(F):
		detF = (
			F11 * (F22*F33 - F23*F32)
		- F12 * (F21*F33 - F23*F31)
		+ F13 * (F21*F32 - F22*F31)
		)
		I3 = detF**2

		# 3) Now form the modified invariants K1, K2, K3:

		#    J  = sqrt(I3)
		J = torch.sqrt(I3)

		#    K1 = I1 * I3^(–1/3) – 3.0
		K1 = I1 * torch.pow(I3, -1.0/3.0) - 3.0

		#    K2 = I2 * I3^(–2/3) – 3.0
		K2 = I2 * torch.pow(I3, -2.0/3.0) - 3.0

		#    K3 = (J - 1.0)^2
		K3 = (J - 1.0)**2


		# Concatenate feature
		x_input = torch.cat((K1,K2,K3,Xc,Yc,Zc),dim=1).float()

		z = x_input.clone()
		z = self.layers[str(0)](z)

		for layer in range(1,self.depth):
			skip = self.skip_layers[str(layer)](x_input)
			z = self.layers[str(layer)](z)
			z += skip
			z = torch.nn.functional.softplus(z)
			if c.use_sftpSquared:
				z = c.scaling_sftpSq*torch.square(z)
			if self.training:
				if self.dropout:
					z = torch.nn.functional.dropout(z,p=self.p_dropout)
		y = self.layers[str(self.depth)](z) + self.skip_layers[str(self.depth)](x_input)
		return y
	


class ICNN3D_global_Taylor(torch.nn.Module):
	"""

	Material model based on Input convex neural network.

	Initialize:
	n_input:			Input layer size
	n_hidden:			Hidden layer size / number of neurons
	n_output:			Output layer size
	use_dropout:		Activate dropout during training
	dropout_rate:		Dropout probability.
	anisotropy_flag:	{single, double} -> type of fiber families
	fiber_type:			{mirror, general} -> type of fiber arrangement in case of two (or more) fiber families.

	Inputs: 			Deformation gradient in the form: (F11,F12,F21,F22)
	Output: 			NN-based strain energy density (W_NN)

	"""
	def __init__(self, n_input, n_hidden, n_output, use_dropout, dropout_rate, anisotropy_flag=None, fiber_type=None):
		super(ICNN3D_global_Taylor, self).__init__()

		# Create Module dicts for the hidden and skip-connection layers
		self.layers = torch.nn.ModuleDict()
		self.skip_layers = torch.nn.ModuleDict()
		self.depth = len(n_hidden)
		self.dropout = c.use_dropout
		self.p_dropout = c.dropout_rate
		self.anisotropy_flag = anisotropy_flag
		self.fiber_type = fiber_type

		if self.anisotropy_flag is not None:
			self.alpha = torch.nn.Parameter((torch.randn(1,1)))#torch.tensor([0.25])#

		self.layers[str(0)] = torch.nn.Linear(n_input, n_hidden[0]).float()
		# Create create NN with number of elements in n_hidden as depth
		for i in range(1, self.depth):
			self.layers[str(i)] = convexLinear(n_hidden[i-1], n_hidden[i]).float()
			self.skip_layers[str(i)] = torch.nn.Linear(n_input, n_hidden[i]).float()

		self.layers[str(self.depth)] = convexLinear(n_hidden[self.depth-1], n_output).float()
		self.skip_layers[str(self.depth)] = convexLinear(n_input, n_output).float()
		self.global_pooling =torch.nn.AdaptiveAvgPool1d(output_size=1)
	def forward(self, x):
		# Get angle
		if self.anisotropy_flag is not None:
			pi = 3.141592653589732
			if self.anisotropy_flag == 'single':
				alpha = pi*torch.sigmoid(self.alpha)
			elif self.anisotropy_flag == 'double':
				alpha = pi/2*torch.sigmoid(self.alpha)

		#Part of the input is the centroids, to avoid assuming homogenenous materials

		# For clarity, we slice out each component as a (batch_size×1) tensor:
		F11 = x[:, 0:1]  # shape = (B,1)
		F12 = x[:, 1:2]
		F13 = x[:, 2:3]

		F21 = x[:, 3:4]
		F22 = x[:, 4:5]
		F23 = x[:, 5:6]

		F31 = x[:, 6:7]
		F32 = x[:, 7:8]
		F33 = x[:, 8:9]

		#Centroid positions
		Xc  = x[:,  9:10]
		Yc  = x[:, 10:11]
		Zc  = x[:, 11:12]

		# 1) Build the Right Cauchy‐Green tensor C = F^T F, component‐wise:
		#    C11 = F11^2 + F21^2 + F31^2
		C11 = F11**2 + F21**2 + F31**2

		#    C12 = F11·F12 + F21·F22 + F31·F32
		C12 = F11*F12 + F21*F22 + F31*F32

		#    C13 = F11·F13 + F21·F23 + F31·F33
		C13 = F11*F13 + F21*F23 + F31*F33

		#    C22 = F12^2 + F22^2 + F32^2
		C22 = F12**2 + F22**2 + F32**2

		#    C23 = F12·F13 + F22·F23 + F32·F33
		C23 = F12*F13 + F22*F23 + F32*F33

		#    C33 = F13^2 + F23^2 + F33^2
		C33 = F13**2 + F23**2 + F33**2

		# (Note: C21 = C12, C31 = C13, C32 = C23, but we only need them for invariants.)

		# 2) Compute the three invariants of C:

		#   I1 = trace(C) = C11 + C22 + C33
		I1 = C11 + C22 + C33

		#   I2 = sum of principal 2×2 minors of C:
		#        I2 =  C11·C22 + C11·C33 + C22·C33  –  (C12^2 + C13^2 + C23^2)
		I2 = (C11 * C22) + (C11 * C33) + (C22 * C33) - (C12**2 + C13**2 + C23**2)

		#   I3 = det(C).  But det(C) = (det F)².  So first form det(F):
		detF = (
			F11 * (F22*F33 - F23*F32)
		- F12 * (F21*F33 - F23*F31)
		+ F13 * (F21*F32 - F22*F31)
		)
		I3 = detF**2

		# 3) Now form the modified invariants K1, K2, K3:

		#    J  = sqrt(I3)
		J = torch.sqrt(I3)

		#    K1 = I1 * I3^(–1/3) – 3.0
		K1 = I1 * torch.pow(I3, -1.0/3.0) - 3.0

		#    K2 = I2 * I3^(–2/3) – 3.0
		K2 = I2 * torch.pow(I3, -2.0/3.0) - 3.0

		#    K3 = (J - 1.0)^2
		K3 = (J - 1.0)**2


		# Concatenate feature
		x_input = torch.cat((K1,K2,K3,Xc,Yc,Zc),dim=1).float()

		z = x_input.clone()
		z = self.layers[str(0)](z)

		for layer in range(1,self.depth):
			skip = self.skip_layers[str(layer)](x_input)
			z = self.layers[str(layer)](z)
			z += skip
			z = torch.nn.functional.softplus(z)
			if c.use_sftpSquared:
				z = c.scaling_sftpSq*torch.square(z)
			if self.training:
				if self.dropout:
					z = torch.nn.functional.dropout(z,p=self.p_dropout)
		y = self.layers[str(self.depth)](z) + self.skip_layers[str(self.depth)](x_input)

		z=self.global_pooling(y.transpose(0,1))
		#print(f'Z: {z.shape}')
		
		return z