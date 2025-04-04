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
#from post_process_compare ismport*

import os
import logging
from datetime import datetime
import config

#########Logging############
# Ensure the output directory exists

os.makedirs(output_dir, exist_ok=True)

# Create a log filename with a timestamp, saved in the output directory
log_filename = os.path.join(output_dir, datetime.now().strftime("experiment_%Y%m%d_%H%M%S.log"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Prints to console as well
    ]
)

logging.info("Starting experiment run in main.py.")


logging.info("Logging configuration settings from config.py:")

# List of config attributes to log
attributes_to_log = ['ensemble_size', 'epochs', 'n_input', 'n_output', 'n_hidden', 'output_dir']

logging.info("Logging selected configuration settings from config.py:")
for attr in attributes_to_log:
    if hasattr(config, attr):
        value = getattr(config, attr)
        logging.info(f"{attr} = {value}")
    else:
        logging.warning(f"Attribute {attr} not found in config")

datasets = []

fem_material = sys.argv[1]
noise_level = sys.argv[2]

logging.info(f"The data for training is from {fem_material} with noise level {noise_level} . We are outputting {n_output} parameters.")

loadsteps = []
if 'NeoHookean' in fem_material:
    loadsteps = [10,20,30]
elif 'Ogden' in fem_material:
    loadsteps = [5,10,15,20,25,30]
elif 'GentThomas' in fem_material:
    loadsteps = [10,20,30]
elif 'ArrudaBoyce' in fem_material:
    loadsteps = [5,10,15,20,25,30,35,40,45,50]
elif 'Anisotropy'  in fem_material:
    loadsteps = [5,10,15,20,25,30,35,40]
elif 'Holzapfel' in fem_material:
    loadsteps = [5,10,15,20,25,30,35,40]
else:
    loadsteps = [10,20,30,40,50,60,70,80]

logging.info(f"fem_material: {fem_material}")
logging.info(f"noise_level: {noise_level}")
logging.info(f"loadsteps: {loadsteps}")

#=====================================================================
# DATA:
#=====================================================================
for loadstep in loadsteps:

    data_path = get_data_path(fem_dir, fem_material,
                              noise_level, loadstep)
    
    logging.info(f"Loading data from: {data_path}")

    data = loadFemData(data_path,
                       noiseLevel = additional_noise,
                       noiseType = 'displacement')

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

logging.info("Model architecture:")
logging.info(model)

print_model_arch(model)

if(random_init):
    model.apply(init_weights)
    #logging.info("Initialized model weights randomly.")

# train/load:
#print('\n\n====================================================================')
#print('Beginning training\n')
#print('Training an ensemble of models...\n')
logging.info("====================================================================")
logging.info("Beginning training")
logging.info("Training an ensemble of models...")

params_all_models=[]
for ensemble_iter in range(ensemble_size):
    #print('\nTraining model '+str(ensemble_iter+1)+' out of '+str(ensemble_size)+'.\n')
    logging.info(f"Training model {ensemble_iter+1} out of {ensemble_size}.")

    model, loss_history, params = train_weak(model, datasets, fem_material, noise_level)
    params_all_models.append(params)
    os.makedirs(output_dir+'/'+fem_material+'/',exist_ok=True)
    torch.save(model.state_dict(), output_dir+'/'+fem_material+'/noise='+noise_level+'_ID='+str(ensemble_iter)+'.pth')
    #logging.info(f"Saved model to {output_dir+'/'+fem_material+'/noise='+noise_level+'_ID='+str(ensemble_iter)+'.pth'}")

    exportList(output_dir+'/'+fem_material+'/','loss_history_noise='+noise_level+'_ID='+str(ensemble_iter),loss_history)
    #logging.info(f"Exported loss history for model {ensemble_iter+1}")
    
    model.apply(init_weights)
    if model.anisotropy_flag is not None:
        model.alpha = torch.nn.Parameter(torch.randn(1,1))
    #logging.info(f"Completed training for model {ensemble_iter+1}")

#print('\n\n=========================================================================================================================')
#print("Final Estimated Parameters:")
logging.info("=========================================================================================================================")
logging.info("All Estimated Parameters:")
logging.info(params_all_models)

logging.info("=========================================================================================================================")


#print(params_all_models)
#print('Evaluating and plotting ICNN on standard strain paths.')
#evaluate_icnn(model, fem_material, noise_level, plot_quantities, output_dir)
evaluate_icnn(model, fem_material, noise_level, plot_quantities, output_dir, params_all_models)
logging.info("Completed experiment run.")

#evaluate_icnn_against_another(model, fem_material, noise_level, plot_quantities, output_dir, compare_against='Holzapfel')
#evaluate_icnn_against_another(model, fem_material, noise_level, plot_quantities, output_dir, compare_against='Ogden')
#evaluate_icnn_against_another(model, fem_material, noise_level, plot_quantities, output_dir, compare_against='NeoHookean')
#evaluate_icnn_against_another(model, fem_material, noise_level, plot_quantities, output_dir, compare_against='HainesWilson')
#evaluate_icnn_against_another(model, fem_material, noise_level, plot_quantities, output_dir, compare_against='Isihara')



print('Completed.')
#print('\n\n=========================================================================================================================')
#print('Best model:')
#print(model.state_dict())
#print('=========================================================================================================================\n\n')
