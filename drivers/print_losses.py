import sys
import matplotlib
sys.path.insert(0, '../')
from core import *
#config
from config import *
#CUDA
initCUDA(cuda)
#supporting files
from model import *
from helper import *
from matplotlib.ticker import FormatStrFormatter

matplotlib.pyplot.rcParams['font.family'] = 'serif'
matplotlib.pyplot.rcParams['mathtext.fontset'] = 'dejavuserif'

ensemble_size=15
fem_material="Isihara"
noise_level="high"

output_dir = '../results_VFM_15_25'

final_losses = torch.zeros((ensemble_size,1))
for ensemble_iter in range(ensemble_size):
    final_losses[ensemble_iter] = pd.read_csv(output_dir+'/'+fem_material+'/loss_history_noise='+noise_level+'_ID='+str(ensemble_iter)+'.csv', header=None).values[-1][1]

#print(final_losses)
print(f'Average of final losses: {torch.mean(final_losses): .3f}')
#final_losses_ratio = final_losses / torch.min(torch.mean(final_losses))
#num_models_remove = torch.where(final_losses_ratio >= accept_ratio)[0].shape[0]
#num_models_keep = ensemble_size - num_models_remove

#idx_best_models = torch.topk(-final_losses.flatten(),num_models_keep).indices
#idx_worst_models = torch.topk(final_losses.flatten(),num_models_remove).indices