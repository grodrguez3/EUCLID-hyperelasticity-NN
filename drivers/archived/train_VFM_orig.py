from core import *
from config import *
import logging


logging.info(f"USING VFM SCRIPT")
#Functions that are related to the VFs

def compute_virtual_strain(grad_Na, virtual_displacement):
    # grad_Na: Gradients of shape functions, shape [num_nodes_per_element, dim]
    # virtual_displacement: Virtual displacements, shape [num_nodes_per_element, dim]
    xi_star = 0.5 * (torch.einsum('ai,aj->aij', virtual_displacement, grad_Na) +
                     torch.einsum('ai,aj->aji', virtual_displacement, grad_Na))
    return xi_star





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
		print('Incorrect scheduler. Choose between `multistep` and `cyclic`.')


	if model.anisotropy_flag is not None:
		print('--------------------------------------------------------------------------------------------------------')
		print('| epoch x/xxx |   lr    |    loss    |     eqb    |  reaction  |    angle   |')
		print('--------------------------------------------------------------------------------------------------------')
	else:
		print('--------------------------------------------------------------------------------------------------------')
		print('| epoch x/xxx |   lr    |    loss    |     eqb    |	reaction	|  	 vfm  |   ewk	|  	iwk ')
		print('--------------------------------------------------------------------------------------------------------')


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

				#Correct VF
				v_x_star = torch.zeros_like(data.x_nodes[:,1]) #data.x_nodes[:,1]
				v_y_star = data.x_nodes[:,1]*2 #torch.sin(np.pi * data.x_nodes[:,1]*0.5) 

				virtual_displacement = torch.stack([v_x_star, v_y_star], dim=1)  #torch.Size([1441, 2])

				#Calculate gradient of virtual displacement
				#gradient_virtual_displacement = torch.stack([v_x_star, torch.cos(np.pi * data.x_nodes[:,1]*0.5)*np.pi*0.5 ], dim=1)  #torch.Size([1441, 2])
				
				#Correct VF
				gradient_virtual_displacement = torch.stack([v_x_star,v_x_star, torch.ones_like(v_y_star)*2 , v_x_star ], dim=1)  #torch.Size([1441, 4])
				
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
					element_evf = gradient_virtual_displacement[data.connectivity[a]]  
					#print(P.shape)
					#print(element_evf.shape)
					external_virtual_work=(P * element_evf).sum(dim=1)* data.qpWeights
					#(external_virtual_work.shape)
					ewk.index_add_(0,data.connectivity[a],external_virtual_work)

					for i in range(dim): # dim is 2 	
						for j in range(dim): #dim is 2

							# # Mapping from **nodes to elements**
							#element_evf = gradient_virtual_displacement[data.connectivity[a]]  	

							#external_virtual_work=P[:,voigt_map[i][j]] *element_evf[:,j]* data.qpWeights # Shape [2752]. I think qpweights is area, and because it is a planar surface it is also volume
							
							#force = P[:,voigt_map[i][j]] * data.gradNa[a][:,j] * data.qpWeights #torch.Size([2752])
							#print(P[:,voigt_map[i][j]].shape)
							force = P[:,voigt_map[i][j]] * element_evf [:,voigt_map[i][j]]* data.qpWeights #torch.Size([2752])
							

							# # Mapping from **elements to nodes**
							f_int_nodes[:,i].index_add_(0,data.connectivity[a],force)
							#ewk[:,i].index_add_(0,data.connectivity[a],external_virtual_work)

							
							for reaction in data.reactions:
								# # Mapping from **nodes to elements**
								element_ivf = virtual_displacement[data.connectivity[a]] 

								#internal_virtual_work+=reaction.force*element_ivf[:,j]* data.qpWeights
								internal_virtual_work+=reaction.force*element_ivf[:,1]* data.qpWeights

							#We only know in boundary
							#internal_virtual_work[~reaction.dofs]=0
							# # Mapping from **elements to nodes**
							#iwk[:,i].index_add_(0,data.connectivity[a],internal_virtual_work)
							iwk.index_add_(0,data.connectivity[a],internal_virtual_work)	
							#print(iwk.shape)
							#print(reaction.dofs.shape)
	
							iwk[~reaction.dofs[:,1]]=0



				# clone f_int_nodes
				f_int_nodes_clone = f_int_nodes.clone()
				# set force on Dirichlet BC nodes to zero
				f_int_nodes_clone[data.dirichlet_nodes] = 0.
				# loss for force equillibrium
				eqb_loss = torch.sum(f_int_nodes_clone**2)


				reaction_loss = torch.tensor([0.])
				for reaction in data.reactions:
					reaction_loss += (torch.sum(f_int_nodes[reaction.dofs]) - reaction.force)**2

				#print(f'EVW:{torch.sum(ewk)}')
				#print(f'IVW:{torch.sum(iwk)}')
				vf_loss=torch.sum(ewk-iwk)**2



				return  eqb_loss, reaction_loss, vf_loss, torch.sum(ewk),torch.sum(iwk), params

			# Compute loss for each displacement snapshot in dataset and add them together
			for data in datasets: #per loading step
				eqb_loss, reaction_loss, vf_loss, ewk,iwk , params= computeLosses(data, model)
				#vf_loss, ewk,iwk = computeLosses(data, model)
				#print(f'VF loss:{vf_loss}')
				#print(f'eqb_loss loss:{eqb_loss}')
				#print(f'reaction_loss loss:{reaction_loss}')

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
					print('| epoch %d/%d | %.1E | %.4E | %.4E | %.4E | %5.6f' % (
						epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item(),torch.sigmoid(model.alpha)*90))
				elif model.anisotropy_flag == 'single':
					print('| epoch %d/%d | %.1E | %.4E | %.4E | %.4E | %5.6f' % (
						epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item(),torch.sigmoid(model.alpha)*180))
			else:
				print('| epoch %d/%d | %.1E | %.4E | %.4E  | %.4E | %.4E | %.4E | %.4E  ' % (
					epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item(), vf_loss.item(), ewk.item(), iwk.item()))

			loss_history.append([epoch_iter+1,loss.item()])

	return model, loss_history, params
