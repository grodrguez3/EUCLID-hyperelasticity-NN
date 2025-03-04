#=====================================================================
# INITIALIZATIONS:
#=====================================================================
#sys and core
import sys
sys.path.insert(0, '../')
from core import *
#config
from config import *
#CUDA
initCUDA(cuda)
#supporting files
from model2 import *
from train import *
#from train import *
from helper import *
from post_process_param import *
#from post_process_compare ismport *

#Modified script to train each network on several samples


datasets = []

fem_materials = sys.argv[1:-1]
noise_level = sys.argv[-1]

# Dictionary to store FEM materials and their corresponding load steps
material_loadsteps = {}

for fem_material in fem_materials:
    if 'NeoHookean' in fem_material:
        material_loadsteps[fem_material] = [10, 20, 30]
    elif 'Ogden' in fem_material:
        material_loadsteps[fem_material] = [5, 10, 15, 20, 25, 30]
    elif 'GentThomas' in fem_material:
        material_loadsteps[fem_material] = [10, 20, 30]
    elif 'ArrudaBoyce' in fem_material:
        material_loadsteps[fem_material] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    elif 'Anisotropy' in fem_material:
        material_loadsteps[fem_material] = [5, 10, 15, 20, 25, 30, 35, 40]
    elif 'Holzapfel' in fem_material:
        material_loadsteps[fem_material] = [5, 10, 15, 20, 25, 30, 35, 40]
    else:
        material_loadsteps[fem_material] = [10, 20, 30, 40, 50, 60, 70, 80]


#=====================================================================
# DATA:
#=====================================================================
datasets = []

for fem_material, loadsteps in material_loadsteps.items():
    for loadstep in loadsteps:
        data_path = get_data_path(fem_dir, fem_material, noise_level, loadstep)

        data = loadFemData(data_path,
                           noiseLevel=additional_noise,
                           noiseType='displacement')

        datasets.append(data)

if 'Holzapfel' in fem_material:
    model = ICNN(n_input=n_input+2,
                 n_hidden=n_hidden,
                 n_output=n_output,
                 use_dropout=use_dropout,
                 dropout_rate=dropout_rate,
                 anisotropy_flag='double',
                 fiber_type='mirror')
elif 'Anisotropy' in fem_material:
    model = ICNN(n_input=n_input+1,
                 n_hidden=n_hidden,
                 n_output=n_output,
                 use_dropout=use_dropout,
                 dropout_rate=dropout_rate,
                 anisotropy_flag='single')
else:
    model = ICNN3(n_input=n_input,
                 n_hidden=n_hidden,
                 n_output=n_output,
                 use_dropout=use_dropout,
                 dropout_rate=dropout_rate)


print_model_arch(model)

if(random_init):
    model.apply(init_weights)

# train/load:
print('\n\n====================================================================')
print('Beginning training\n')
print('Training an ensemble of models...\n')
params_all_models=[]
for ensemble_iter in range(ensemble_size):
    print('\nTraining model '+str(ensemble_iter+1)+' out of '+str(ensemble_size)+'.\n')
    
    for fem_material in fem_materials: #Check that the model not always expects the same order. I think it does not matter because each is a new model. 
 
        model, loss_history, params = train_weak(model, datasets, fem_material, noise_level)
        params_all_models.append(params)
        os.makedirs(output_dir+'/'+fem_material+'/',exist_ok=True)
        torch.save(model.state_dict(), output_dir+'/'+fem_material+'/noise='+noise_level+'_ID='+str(ensemble_iter)+'.pth')
        exportList(output_dir+'/'+fem_material+'/','loss_history_noise='+noise_level+'_ID='+str(ensemble_iter),loss_history)
    
    model.apply(init_weights)
    if model.anisotropy_flag is not None:
        model.alpha = torch.nn.Parameter(torch.randn(1,1))
print('\n\n=========================================================================================================================')
print("Final Estimated Parameters:")
#print(params_all_models)
#print('Evaluating and plotting ICNN on standard strain paths.')
#evaluate_icnn(model, fem_material, noise_level, plot_quantities, output_dir)
evaluate_icnn(model, fem_material, noise_level, plot_quantities, output_dir, params_all_models)
#evaluate_icnn_against_another(model, fem_material, noise_level, plot_quantities, output_dir, compare_against='Holzapfel')
#evaluate_icnn_against_another(model, fem_material, noise_level, plot_quantities, output_dir, compare_against='Ogden')
#evaluate_icnn_against_another(model, fem_material, noise_level, plot_quantities, output_dir, compare_against='NeoHookean')
#evaluate_icnn_against_another(model, fem_material, noise_level, plot_quantities, output_dir, compare_against='HainesWilson')
#evaluate_icnn_against_another(model, fem_material, noise_level, plot_quantities, output_dir, compare_against='Isihara')



print('Completed.')
print('\n\n=========================================================================================================================')
print('Best model:')
#print(model.state_dict())
print('=========================================================================================================================\n\n')
