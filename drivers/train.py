from core import *
from config import *
import logging


def train_weak(model, datasets, fem_material, noise_level):

	#loss history
	loss_history = []

	# optimizer
	if(opt_method == 'adam'):
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	elif(opt_method == 'lbfgs'):
		optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, line_search_fn='strong_wolfe')
	elif(opt_method == 'sgd'):
		optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	elif(opt_method == 'rmsprop'):
		optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
	else:
		raise ValueError('Incorrect choice of optimizer')

	#scheduler
	if lr_schedule == 'multistep':
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,lr_milestones,lr_decay,last_epoch=-1)
	elif lr_schedule == 'cyclic':
		scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, cycle_momentum=cycle_momentum, step_size_up=step_size_up, step_size_down=step_size_down)
	else:
		logging.info('Incorrect scheduler. Choose between `multistep` and `cyclic`.')


	if model.anisotropy_flag is not None:
		logging.info('--------------------------------------------------------------------------------------------------------')
		logging.info('| epoch x/xxx |   lr    |    loss    |     eqb    |  reaction  |    angle   |')
		logging.info('--------------------------------------------------------------------------------------------------------')
	else:
		logging.info('--------------------------------------------------------------------------------------------------------')
		logging.info('| epoch x/xxx |   lr    |    loss    |     eqb    |  reaction  |')
		logging.info('--------------------------------------------------------------------------------------------------------')


	for epoch_iter in range(epochs):

		def closure():

			optimizer.zero_grad()

			loss = torch.tensor([0.])

			def computeLosses(data, model):

				# Zero deformation gradient dummy data
				F_0 = torch.zeros((1,4))
				F_0[:,0] = 1
				F_0[:,3] = 1

				F11_0 = F_0[:,0:1]
				F12_0 = F_0[:,1:2]
				F21_0 = F_0[:,2:3]
				F22_0 = F_0[:,3:4]

				F11_0.requires_grad = True
				F12_0.requires_grad = True
				F21_0.requires_grad = True
				F22_0.requires_grad = True

				# Get components of F from dataset
				F11 = data.F[:,0:1]
				F12 = data.F[:,1:2]
				F21 = data.F[:,2:3]
				F22 = data.F[:,3:4]

				# Allow to build computational graph
				F11.requires_grad = True
				F12.requires_grad = True
				F21.requires_grad = True
				F22.requires_grad = True

				# Forward pass NN to obtain W_NN
				#logging.info("Estimation of W_NN")
				W_NN, params = model(torch.cat((F11,F12,F21,F22),dim=1)) #shape of torch.Size([2752, 1])

				# Get gradients of W w.r.t F
				dW_NN_dF11 = torch.autograd.grad(W_NN,F11,torch.ones(F11.shape[0],1),create_graph=True)[0]
				dW_NN_dF12 = torch.autograd.grad(W_NN,F12,torch.ones(F12.shape[0],1),create_graph=True)[0]
				dW_NN_dF21 = torch.autograd.grad(W_NN,F21,torch.ones(F21.shape[0],1),create_graph=True)[0]
				dW_NN_dF22 = torch.autograd.grad(W_NN,F22,torch.ones(F22.shape[0],1),create_graph=True)[0]

				# Assemble First Piola-Kirchhoff stress components
				P_NN = torch.cat((dW_NN_dF11,dW_NN_dF12,dW_NN_dF21,dW_NN_dF22),dim=1)

				# Forward pass to obtain zero deformation energy correction
				#logging.info("Estimation of W_NN_0")
				W_NN_0, _ = model(torch.cat((F11_0,F12_0,F21_0,F22_0),dim=1)) #shape of torch.Size([1, 1])

				# Get gradients of W_NN_0 w.r.t F
				dW_NN_dF11_0 = torch.autograd.grad(W_NN_0,F11_0,torch.ones(F11_0.shape[0],1),create_graph=True)[0]
				dW_NN_dF12_0 = torch.autograd.grad(W_NN_0,F12_0,torch.ones(F12_0.shape[0],1),create_graph=True)[0]
				dW_NN_dF21_0 = torch.autograd.grad(W_NN_0,F21_0,torch.ones(F21_0.shape[0],1),create_graph=True)[0]
				dW_NN_dF22_0 = torch.autograd.grad(W_NN_0,F22_0,torch.ones(F22_0.shape[0],1),create_graph=True)[0]

				# Get stress at zero deformation
				P_NN_0 = torch.cat((dW_NN_dF11_0,dW_NN_dF12_0,dW_NN_dF21_0,dW_NN_dF22_0),dim=1)

				# Initialize zero stress correction term
				P_cor = torch.zeros_like(P_NN)

				# Compute stress correction components according to Ansatz
				P_cor[:,0:1] = F11*-P_NN_0[:,0:1] + F12*-P_NN_0[:,2:3]
				P_cor[:,1:2] = F11*-P_NN_0[:,1:2] + F12*-P_NN_0[:,3:4]
				P_cor[:,2:3] = F21*-P_NN_0[:,0:1] + F22*-P_NN_0[:,2:3]
				P_cor[:,3:4] = F21*-P_NN_0[:,1:2] + F22*-P_NN_0[:,3:4]

				# Compute final stress (NN + correction)
				P = P_NN + P_cor


				# compute internal forces on nodes
				f_int_nodes = torch.zeros(data.numNodes,dim)
				for a in range(num_nodes_per_element):
					for i in range(dim):
						for j in range(dim):
							force = P[:,voigt_map[i][j]] * data.gradNa[a][:,j] * data.qpWeights #element
							f_int_nodes[:,i].index_add_(0,data.connectivity[a],force) #node

				# clone f_int_nodes
				f_int_nodes_clone = f_int_nodes.clone()
				# set force on Dirichlet BC nodes to zero
				f_int_nodes_clone[data.dirichlet_nodes] = 0.
				# loss for force equillibrium
				eqb_loss = torch.sum(f_int_nodes_clone**2)

				reaction_loss = torch.tensor([0.])
				for reaction in data.reactions:
					reaction_loss += (torch.sum(f_int_nodes[reaction.dofs]) - reaction.force)**2

				return  eqb_loss, reaction_loss, params

			# Compute loss for each displacement snapshot in dataset and add them together
			for data in datasets:
				eqb_loss, reaction_loss, params = computeLosses(data, model)
				loss += eqb_loss_factor * eqb_loss + reaction_loss_factor * reaction_loss

			# back propagate
			loss.backward()

			return loss, eqb_loss, reaction_loss, params

		loss, eqb_loss, reaction_loss, params = optimizer.step(closure)
		scheduler.step()
		#if epoch_iter==25:
		#	logging.info(params)


		if(epoch_iter % verbose_frequency == 0):
			if model.anisotropy_flag is not None:
				if model.anisotropy_flag == 'double':
					logging.info('| epoch %d/%d | %.1E | %.4E | %.4E | %.4E | %5.6f' % (
						epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item(),torch.sigmoid(model.alpha)*90))
				elif model.anisotropy_flag == 'single':
					logging.info('| epoch %d/%d | %.1E | %.4E | %.4E | %.4E | %5.6f' % (
						epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item(),torch.sigmoid(model.alpha)*180))
			else:
				logging.info('| epoch %d/%d | %.1E | %.4E | %.4E | %.4E' % (
					epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item()))

			
			loss_history.append([epoch_iter+1,loss.item()])

	return model, loss_history, params

def compute_virtual_strain(grad_Na, virtual_displacement):
    # grad_Na: Gradients of shape functions, shape [num_nodes_per_element, dim]
    # virtual_displacement: Virtual displacements, shape [num_nodes_per_element, dim]
    xi_star = 0.5 * (torch.einsum('ai,aj->aij', virtual_displacement, grad_Na) +
                     torch.einsum('ai,aj->aji', virtual_displacement, grad_Na))
    return xi_star





def train_weak_VFM(model, datasets, fem_material, noise_level):
	logging.info("Using VFM regularization")
	#loss history
	loss_history = []

	# optimizer
	if(opt_method == 'adam'):
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	elif(opt_method == 'lbfgs'):
		optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, line_search_fn='strong_wolfe')
	elif(opt_method == 'sgd'):
		optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	elif(opt_method == 'rmsprop'):
		optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
	else:
		raise ValueError('Incorrect choice of optimizer')

	#scheduler
	if lr_schedule == 'multistep':
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,lr_milestones,lr_decay,last_epoch=-1)
	elif lr_schedule == 'cyclic':
		scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, cycle_momentum=cycle_momentum, step_size_up=step_size_up, step_size_down=step_size_down)
	else:
		logging.info('Incorrect scheduler. Choose between `multistep` and `cyclic`.')


	if model.anisotropy_flag is not None:
		logging.info('--------------------------------------------------------------------------------------------------------')
		logging.info('| epoch x/xxx |   lr    |    loss    |     eqb    |  reaction  |    angle   |')
		logging.info('--------------------------------------------------------------------------------------------------------')
	else:
		
		logging.info('--------------------------------------------------------------------------------------------------------')
		logging.info('| epoch x/xxx |   lr   |    loss    |     eqb     |   reaction  |    vfm    |     ewk     |     iwk    |')
		logging.info('--------------------------------------------------------------------------------------------------------')


	for epoch_iter in range(epochs):

		def closure():

			optimizer.zero_grad()

			loss = torch.tensor([0.])

			def computeLosses(data, model):

				# Zero deformation gradient dummy data
				F_0 = torch.zeros((1,4))
				F_0[:,0] = 1
				F_0[:,3] = 1

				F11_0 = F_0[:,0:1]
				F12_0 = F_0[:,1:2]
				F21_0 = F_0[:,2:3]
				F22_0 = F_0[:,3:4]

				F11_0.requires_grad = True
				F12_0.requires_grad = True
				F21_0.requires_grad = True
				F22_0.requires_grad = True

				# Get components of F from dataset
				F11 = data.F[:,0:1]
				F12 = data.F[:,1:2]
				F21 = data.F[:,2:3]
				F22 = data.F[:,3:4]

				# Allow to build computational graph
				F11.requires_grad = True
				F12.requires_grad = True
				F21.requires_grad = True
				F22.requires_grad = True

				# Forward pass NN to obtain W_NN
				W_NN, params = model(torch.cat((F11,F12,F21,F22),dim=1))

				# Get gradients of W w.r.t F
				dW_NN_dF11 = torch.autograd.grad(W_NN,F11,torch.ones(F11.shape[0],1),create_graph=True)[0]
				dW_NN_dF12 = torch.autograd.grad(W_NN,F12,torch.ones(F12.shape[0],1),create_graph=True)[0]
				dW_NN_dF21 = torch.autograd.grad(W_NN,F21,torch.ones(F21.shape[0],1),create_graph=True)[0]
				dW_NN_dF22 = torch.autograd.grad(W_NN,F22,torch.ones(F22.shape[0],1),create_graph=True)[0]

				# Assemble First Piola-Kirchhoff stress components
				P_NN = torch.cat((dW_NN_dF11,dW_NN_dF12,dW_NN_dF21,dW_NN_dF22),dim=1)

				# Forward pass to obtain zero deformation energy correction
				W_NN_0, _ = model(torch.cat((F11_0,F12_0,F21_0,F22_0),dim=1))

				# Get gradients of W_NN_0 w.r.t F
				dW_NN_dF11_0 = torch.autograd.grad(W_NN_0,F11_0,torch.ones(F11_0.shape[0],1),create_graph=True)[0]
				dW_NN_dF12_0 = torch.autograd.grad(W_NN_0,F12_0,torch.ones(F12_0.shape[0],1),create_graph=True)[0]
				dW_NN_dF21_0 = torch.autograd.grad(W_NN_0,F21_0,torch.ones(F21_0.shape[0],1),create_graph=True)[0]
				dW_NN_dF22_0 = torch.autograd.grad(W_NN_0,F22_0,torch.ones(F22_0.shape[0],1),create_graph=True)[0]

				# Get stress at zero deformation
				P_NN_0 = torch.cat((dW_NN_dF11_0,dW_NN_dF12_0,dW_NN_dF21_0,dW_NN_dF22_0),dim=1)

				# Initialize zero stress correction term
				P_cor = torch.zeros_like(P_NN)

				# Compute stress correction components according to Ansatz
				P_cor[:,0:1] = F11*-P_NN_0[:,0:1] + F12*-P_NN_0[:,2:3]
				P_cor[:,1:2] = F11*-P_NN_0[:,1:2] + F12*-P_NN_0[:,3:4]
				P_cor[:,2:3] = F21*-P_NN_0[:,0:1] + F22*-P_NN_0[:,2:3]
				P_cor[:,3:4] = F21*-P_NN_0[:,1:2] + F22*-P_NN_0[:,3:4]

				# Compute final stress (NN + correction)
				P = P_NN + P_cor #P.shape:torch.Size([2752, 4]) 

				#energy_loss = torch.abs(torch.mean(P)) #test strong from of eq 

				
				#Define VFs. For this case is an in plane deformation with vx=0, vy=x^2.  
				#Incorrect VF
				#v_x_star = data.x_nodes[:,1]**2 #data.x_nodes[:,1]
				#v_y_star = data.x_nodes[:,1]*2

				#Correct VF. Has the shape of nodes. 
				v_x_star1 = torch.zeros_like(data.x_nodes[:,1]) 
				v_y_star1 = data.y_nodes[:,1]*2 #corrected

				v_x_star2 = data.x_nodes[:,1]/2 
				v_y_star2 = torch.zeros_like(data.x_nodes[:,1]) 


				virtual_displacement1 = torch.stack([v_x_star1, v_y_star1], dim=1)  #torch.Size([1441, 2])
				virtual_displacement2 = torch.stack([v_x_star2, v_y_star2], dim=1)  #torch.Size([1441, 2])

				#Calculate gradient of virtual displacement
				#gradient_virtual_displacement = torch.stack([v_x_star, torch.cos(np.pi * data.x_nodes[:,1]*0.5)*np.pi*0.5 ], dim=1)  #torch.Size([1441, 2])
				
				#Correct VF
				gradient_virtual_displacement1 = torch.stack([v_x_star1,v_x_star1, torch.ones_like(v_y_star1)*2 , v_x_star1 ], dim=1)  #torch.Size([1441, 4])
				gradient_virtual_displacement2 = torch.stack([torch.ones_like(v_y_star1)/2,v_x_star1, v_x_star1 , v_x_star1 ], dim=1)  #torch.Size([1441, 4])

				#Incorrect VF
				#gradient_virtual_displacement = torch.stack([torch.zeros_like(v_y_star),data.x_nodes[:,1], 2*torch.ones_like(v_y_star) , data.x_nodes[:,1] ], dim=1)  #torch.Size([1441, 4])

				num_nodes_per_element = 3  # Triangular elements
				# compute internal forces on nodes
				f_int_nodes = torch.zeros(data.numNodes,dim)
				internal_virtual_work=torch.zeros(data.numElements)

				ewk=torch.zeros(data.numNodes)
				iwk=torch.zeros(data.numNodes)


				for a in range(num_nodes_per_element): #this is 3

					# # Mapping from **nodes to elements**
					element_ivf1 = gradient_virtual_displacement1[data.connectivity[a]]  
					element_ivf2 = gradient_virtual_displacement2[data.connectivity[a]] 

					F=torch.cat((F11,F12,F21,F22),dim=1)
					J = computeJacobian(F)
					correct_internal_virtual_work= ((1/J*P*F)* (element_ivf1+element_ivf2)).sum(dim=1)* data.qpWeights


					#(external_virtual_work.shape)
					iwk.index_add_(0,data.connectivity[a],correct_internal_virtual_work) 

					for i in range(dim): # dim is 2 	
						for j in range(dim): #dim is 2

							# # Mapping from **nodes to elements**
							#element_evf = gradient_virtual_displacement[data.connectivity[a]]  	

							#external_virtual_work=P[:,voigt_map[i][j]] *element_evf[:,j]* data.qpWeights # Shape [2752]. I think qpweights is area, and because it is a planar surface it is also volume
							
							#force = P[:,voigt_map[i][j]] * data.gradNa[a][:,j] * data.qpWeights #torch.Size([2752])
							#logging.info(P[:,voigt_map[i][j]].shape)
							force = P[:,voigt_map[i][j]] * element_evf [:,voigt_map[i][j]]* data.qpWeights #torch.Size([2752])
							

							# # Mapping from **elements to nodes**
							f_int_nodes[:,i].index_add_(0,data.connectivity[a],force)
							#ewk[:,i].index_add_(0,data.connectivity[a],external_virtual_work)

							
							#for reaction in data.reactions:
								# # Mapping from **nodes to elements**
								#element_ivf = virtual_displacement[data.connectivity[a]] 

								#internal_virtual_work+=reaction.force*element_ivf[:,j]* data.qpWeights
								#external_virtual_work+=reaction.force*element_ivf[:,1]* data.qpWeights

							#We only know in boundary
							#internal_virtual_work[~reaction.dofs]=0
							# # Mapping from **elements to nodes**
							#iwk[:,i].index_add_(0,data.connectivity[a],internal_virtual_work)
							#iwk.index_add_(0,data.connectivity[a],internal_virtual_work)	
							#logging.info(iwk.shape)
							#logging.info(reaction.dofs.shape)
	
							#iwk[~reaction.dofs[:,1]]=0



				# clone f_int_nodes
				f_int_nodes_clone = f_int_nodes.clone()
				# set force on Dirichlet BC nodes to zero
				f_int_nodes_clone[data.dirichlet_nodes] = 0.
				# loss for force equillibrium
				eqb_loss = torch.sum(f_int_nodes_clone**2)


				reaction_loss = torch.tensor([0.])
				for reaction in data.reactions:
					reaction_loss += (torch.sum(f_int_nodes[reaction.dofs]) - reaction.force)**2

				#logging.info(f'EVW:{torch.sum(ewk)}')
				#logging.info(f'IVW:{torch.sum(iwk)}')
				vf_loss=torch.sum(ewk-iwk)**2



				return  eqb_loss, reaction_loss, vf_loss, torch.sum(ewk),torch.sum(iwk), params

			# Compute loss for each displacement snapshot in dataset and add them together
			for data in datasets: #per loading step
				eqb_loss, reaction_loss, vf_loss, ewk,iwk , params= computeLosses(data, model)
				#vf_loss, ewk,iwk = computeLosses(data, model)
				#logging.info(f'VF loss:{vf_loss}')
				#logging.info(f'eqb_loss loss:{eqb_loss}')
				#logging.info(f'reaction_loss loss:{reaction_loss}')

				loss += eqb_loss_factor * eqb_loss + reaction_loss_factor * reaction_loss + vf_loss# (VF factor on uncertainty of automatically chosen VF)

			# back propagate
			loss.backward()

			return loss, eqb_loss, reaction_loss, vf_loss, ewk,iwk , params

		loss, eqb_loss, reaction_loss, vf_loss, ewk,iwk , params= optimizer.step(closure)
		#loss,vf_loss, ewk,iwk = optimizer.step(closure)
		scheduler.step()


		if(epoch_iter % verbose_frequency == 0):
			if model.anisotropy_flag is not None:
				if model.anisotropy_flag == 'double':
					logging.info('| epoch %d/%d | %.1E | %.4E | %.4E | %.4E | %5.6f' % (
						epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item(),torch.sigmoid(model.alpha)*90))
				elif model.anisotropy_flag == 'single':
					logging.info('| epoch %d/%d | %.1E | %.4E | %.4E | %.4E | %5.6f' % (
						epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item(),torch.sigmoid(model.alpha)*180))
			else:
				logging.info('| epoch %d/%d | %.1E | %.4E | %.4E  | %.4E | %.4E | %.4E | %.4E  ' % (
					epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item(), vf_loss.item(), ewk.item(), iwk.item()))

			loss_history.append([epoch_iter+1,loss.item()])

	return model, loss_history, params





def train_weak_VFM_correct(model, datasets, fem_material, noise_level):

	#loss history
	loss_history = []

	# optimizer
	if(opt_method == 'adam'):
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	elif(opt_method == 'lbfgs'):
		optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, line_search_fn='strong_wolfe')
	elif(opt_method == 'sgd'):
		optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	elif(opt_method == 'rmsprop'):
		optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
	else:
		raise ValueError('Incorrect choice of optimizer')

	#scheduler
	if lr_schedule == 'multistep':
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,lr_milestones,lr_decay,last_epoch=-1)
	elif lr_schedule == 'cyclic':
		scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, cycle_momentum=cycle_momentum, step_size_up=step_size_up, step_size_down=step_size_down)
	else:
		print('Incorrect scheduler. Choose between `multistep` and `cyclic`.')


	if model.anisotropy_flag is not None:
		print('--------------------------------------------------------------------------------------------------------')
		print('| epoch x/xxx |   lr    |    loss    |     eqb    |  reaction  |    angle   |')
		print('--------------------------------------------------------------------------------------------------------')
	else:
		logging.info('--------------------------------------------------------------------------------------------------------')
		logging.info('| epoch x/xxx |   lr   |    loss    |     eqb     |   reaction  |    vfm    |     ewk     |     iwk    |')
		logging.info('--------------------------------------------------------------------------------------------------------')


	for epoch_iter in range(epochs):

		def closure():

			optimizer.zero_grad()

			loss = torch.tensor([0.])

			def computeLosses(data, model):

				# Zero deformation gradient dummy data
				F_0 = torch.zeros((1,4))
				F_0[:,0] = 1
				F_0[:,3] = 1

				F11_0 = F_0[:,0:1]
				F12_0 = F_0[:,1:2]
				F21_0 = F_0[:,2:3]
				F22_0 = F_0[:,3:4]

				F11_0.requires_grad = True
				F12_0.requires_grad = True
				F21_0.requires_grad = True
				F22_0.requires_grad = True

				# Get components of F from dataset
				F11 = data.F[:,0:1]
				F12 = data.F[:,1:2]
				F21 = data.F[:,2:3]
				F22 = data.F[:,3:4]

				# Allow to build computational graph
				F11.requires_grad = True
				F12.requires_grad = True
				F21.requires_grad = True
				F22.requires_grad = True

				# Assemble deformation gradient and Jacobian
				F = torch.cat((F11,F12,F21,F22),dim=1) #H
				J = computeJacobian(F)

				# Forward pass NN to obtain W_NN
				W_NN = model(torch.cat((F11,F12,F21,F22),dim=1))

				# Get gradients of W w.r.t F
				dW_NN_dF11 = torch.autograd.grad(W_NN,F11,torch.ones(F11.shape[0],1),create_graph=True)[0]
				dW_NN_dF12 = torch.autograd.grad(W_NN,F12,torch.ones(F12.shape[0],1),create_graph=True)[0]
				dW_NN_dF21 = torch.autograd.grad(W_NN,F21,torch.ones(F21.shape[0],1),create_graph=True)[0]
				dW_NN_dF22 = torch.autograd.grad(W_NN,F22,torch.ones(F22.shape[0],1),create_graph=True)[0]

				# Assemble First Piola-Kirchhoff stress components
				P_NN = torch.cat((dW_NN_dF11,dW_NN_dF12,dW_NN_dF21,dW_NN_dF22),dim=1)

				# Forward pass to obtain zero deformation energy correction
				W_NN_0 = model(torch.cat((F11_0,F12_0,F21_0,F22_0),dim=1))

				# Get gradients of W_NN_0 w.r.t F
				dW_NN_dF11_0 = torch.autograd.grad(W_NN_0,F11_0,torch.ones(F11_0.shape[0],1),create_graph=True)[0]
				dW_NN_dF12_0 = torch.autograd.grad(W_NN_0,F12_0,torch.ones(F12_0.shape[0],1),create_graph=True)[0]
				dW_NN_dF21_0 = torch.autograd.grad(W_NN_0,F21_0,torch.ones(F21_0.shape[0],1),create_graph=True)[0]
				dW_NN_dF22_0 = torch.autograd.grad(W_NN_0,F22_0,torch.ones(F22_0.shape[0],1),create_graph=True)[0]

				# Get stress at zero deformation
				P_NN_0 = torch.cat((dW_NN_dF11_0,dW_NN_dF12_0,dW_NN_dF21_0,dW_NN_dF22_0),dim=1)

				# Initialize zero stress correction term
				P_cor = torch.zeros_like(P_NN)

				# Compute stress correction components according to Ansatz
				P_cor[:,0:1] = F11*-P_NN_0[:,0:1] + F12*-P_NN_0[:,2:3]
				P_cor[:,1:2] = F11*-P_NN_0[:,1:2] + F12*-P_NN_0[:,3:4]
				P_cor[:,2:3] = F21*-P_NN_0[:,0:1] + F22*-P_NN_0[:,2:3]
				P_cor[:,3:4] = F21*-P_NN_0[:,1:2] + F22*-P_NN_0[:,3:4]

				# Compute final stress (NN + correction)
				P = P_NN + P_cor

				#Define VFs. For this case is an in plane deformation with vx=0, vy=x^2.  
				#Incorrect VF
				#v_x_star = data.x_nodes[:,1]**2 #data.x_nodes[:,1]
				#v_y_star = data.x_nodes[:,1]*2

				#Correct VF. Has the shape of nodes. 
				virtual_displacement = {
					'x': [[] for _ in range(2)],   # creates [[], []]
					'y': [[] for _ in range(2)]}   # creates [[], []]


				v_x_star1 = torch.zeros_like(data.x_nodes[:,1]) 
				v_y_star1 = data.u_nodes[:,1]*2 #corrected

				v_x_star2 = data.u_nodes[:,1]/2 
				v_y_star2 = torch.zeros_like(data.x_nodes[:,1]) 


				virtual_displacement1 = torch.stack([v_x_star1, v_y_star1], dim=1)  #torch.Size([1441, 2])
				virtual_displacement2 = torch.stack([v_x_star2, v_y_star2], dim=1)  #torch.Size([1441, 2])

				#Calculate gradient of virtual displacement
				#gradient_virtual_displacement = torch.stack([v_x_star, torch.cos(np.pi * data.x_nodes[:,1]*0.5)*np.pi*0.5 ], dim=1)  #torch.Size([1441, 2])
				
				#Correct VF
				gradient_virtual_displacement1 = torch.stack([v_x_star1,v_x_star1, torch.ones_like(v_y_star1)*2 , v_x_star1 ], dim=1)  #torch.Size([1441, 4])
				gradient_virtual_displacement2 = torch.stack([torch.ones_like(v_y_star1)/2,v_x_star1, v_x_star1 , v_x_star1 ], dim=1)  #torch.Size([1441, 4])

				#Incorrect VF
				#gradient_virtual_displacement = torch.stack([torch.zeros_like(v_y_star),data.x_nodes[:,1], 2*torch.ones_like(v_y_star) , data.x_nodes[:,1] ], dim=1)  #torch.Size([1441, 4])
				ewk=torch.zeros(data.numNodes)
				iwk=torch.zeros(data.numNodes)

				#Compute internal work
				for a in range(num_nodes_per_element): #this is 3

					# # Mapping from **nodes to elements**
					element_ivf1 = gradient_virtual_displacement1[data.connectivity[a]]  
					element_ivf2 = gradient_virtual_displacement2[data.connectivity[a]] 

					#F=torch.cat((F11,F12,F21,F22),dim=1)
					#F_t=torch.cat((F11,F21,F12,F22),dim=1)
					F_mat = torch.stack([
						torch.stack([F11, F12], dim=-1),
						torch.stack([F21, F22], dim=-1)
					], dim=-2).squeeze() #torch.Size([2752, 2, 2]) after squeeze                  # [nElem,2,2]

					P_mat = torch.stack([
						torch.stack([P[:,0:1], P[:,1:2]], dim=-1),
						torch.stack([P[:,2:3], P[:,3:4]], dim=-1)
					], dim=-2).squeeze()                   # [nElem,2,2]
					#print(P_mat.shape)	
					# --- form Cauchy stress: σ = (1/J) P · Fᵀ ---
					J = torch.det(F_mat)         # [nElem]

					sigma = (P_mat @ F_mat.transpose(-2,-1)) / J.view(-1,1,1) # [nElem,2,2]
												
					# --- flatten σᵢⱼ in the standard row‐major order [11,12,21,22] ---
					sigma_flat = sigma.view(-1,4)   # [nElem,4]	
					#print(sigma_flat.shape)				
					correct_internal_virtual_work= (sigma_flat* (element_ivf1+element_ivf2)).sum(dim=1)* data.qpWeights

					iwk.index_add_(0,data.connectivity[a],correct_internal_virtual_work) #Nodes 


				#Compute external work in elements:
				c=-1
				for reaction in data.reactions:
					
					#Reaction.dofs is 1441,2 and has True, False values. 
					# if *any* component of this reaction is negative, skip it entirely
					if (reaction.force < 0): #ok
						#print(reaction.force)
						continue
					else:
						c+=1
						for a in range(num_nodes_per_element):
							#print(f'React dofs shape: {[reaction.dofs.shape]}') #1441, 2
							if c==0:
								#print('VF 1') #assume x force goes first
								
								element_virtual_displacement = virtual_displacement2[data.connectivity[a]]

							else:
								#print('VF 2') #assume y force goes after
								element_virtual_displacement = virtual_displacement1[data.connectivity[a]]

							
							element_reaction_dofs = reaction.dofs[data.connectivity[a]].int()  #in 0 and 1

							#print(f'Reaction dofs{element_reaction_dofs}')
			
							node_external_virtual_work = torch.sum(element_virtual_displacement*element_reaction_dofs, dim=1)*reaction.force #Ask
							
							element_virtual_work=node_external_virtual_work[data.connectivity[a]]* data.qpWeights
							
							ewk.index_add_(0,data.connectivity[a],element_virtual_work) #Nodes 


				# compute internal forces on nodes
				f_int_nodes = torch.zeros(data.numNodes,dim)
				for a in range(num_nodes_per_element):
					for i in range(dim):
						for j in range(dim):
							force = P[:,voigt_map[i][j]] * data.gradNa[a][:,j] * data.qpWeights #per element
							#print(f'force shape{force.shape}')
							#print(f'force shape{force}')
							f_int_nodes[:,i].index_add_(0,data.connectivity[a],force) #per node
							#print(f'f_int_nodes{f_int_nodes[:,i]}')

				# clone f_int_nodes
				f_int_nodes_clone = f_int_nodes.clone()
				# set force on Dirichlet BC nodes to zero
				f_int_nodes_clone[data.dirichlet_nodes] = 0.
				# loss for force equillibrium
				eqb_loss = torch.sum(f_int_nodes_clone**2)

				reaction_loss = torch.tensor([0.])
				for reaction in data.reactions:
					#Reaction.dofs is 1441,2 and has True, False values. 
					reaction_loss += (torch.sum(f_int_nodes[reaction.dofs]) - reaction.force)**2
					#print(f'reaction.force{reaction.force}')
				
				#logging.info(f'EVW:{torch.sum(ewk)}')
				#logging.info(f'IVW:{torch.sum(iwk)}')
				vf_loss=torch.sum(ewk-iwk)**2



				return  eqb_loss, reaction_loss, vf_loss, torch.sum(ewk),torch.sum(iwk)





			# Compute loss for each displacement snapshot in dataset and add them together
			for data in datasets:
				eqb_loss, reaction_loss, vf_loss, ewk,iwk = computeLosses(data, model)
				loss += eqb_loss_factor * eqb_loss + reaction_loss_factor * reaction_loss +vf_loss

			# back propagate
			loss.backward()

			return loss, eqb_loss, reaction_loss, vf_loss, ewk, iwk

		loss, eqb_loss, reaction_loss, vf_loss, ewk, iwk = optimizer.step(closure)

		scheduler.step()


		if(epoch_iter % verbose_frequency == 0):
			if model.anisotropy_flag is not None:
				if model.anisotropy_flag == 'double':
					print('| epoch %d/%d | %.1E | %.4E | %.4E | %.4E | %5.6f' % (
						epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item(),torch.sigmoid(model.alpha)*90))
				elif model.anisotropy_flag == 'single':
					print('| epoch %d/%d | %.1E | %.4E | %.4E | %.4E | %5.6f' % (
						epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item(),torch.sigmoid(model.alpha)*180))
			else:
				logging.info('| epoch %d/%d | %.1E | %.4E | %.4E  | %.4E | %.4E | %.4E | %.4E  ' % (
					epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item(), vf_loss.item(), ewk.item(), iwk.item()))

			loss_history.append([epoch_iter+1,loss.item()])

	return model, loss_history