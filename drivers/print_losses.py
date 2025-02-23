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
fem_material="NeoHookean"
noise_level="high"
epochs=25
output_dir = '../results_15_25_VFM'


final_losses = torch.zeros((ensemble_size,1))
all_losses = np.zeros((ensemble_size,epochs))

for ensemble_iter in range(ensemble_size):
    final_losses[ensemble_iter] = pd.read_csv(output_dir+'/'+fem_material+'/loss_history_noise='+noise_level+'_ID='+str(ensemble_iter)+'.csv', header=None).values[-1][1]
    all_losses[ensemble_iter,:] = pd.read_csv(output_dir+'/'+fem_material+'/loss_history_noise='+noise_level+'_ID='+str(ensemble_iter)+'.csv', header=None).values[:,1]
#print(all_losses)
print(f'Average of final losses: {torch.mean(final_losses): .3f}')

plt.figure(figsize=(8, 5))

# Plot each model’s loss per epoch
for i in range(ensemble_size):
    if i ==11:
        continue
    plt.plot(all_losses[i,20:], label=f"Model {i}")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Loss History for {fem_material} (noise={noise_level})")
plt.legend()
plt.grid(True)

# ---- Save as PDF ----
plt.savefig(output_dir+"/loss_history.pdf", format="pdf", bbox_inches="tight")

plt.show()
#final_losses_ratio = final_losses / torch.min(torch.mean(final_losses))
#num_models_remove = torch.where(final_losses_ratio >= accept_ratio)[0].shape[0]
#num_models_keep = ensemble_size - num_models_remove

#idx_best_models = torch.topk(-final_losses.flatten(),num_models_keep).indices
#idx_worst_models = torch.topk(final_losses.flatten(),num_models_remove).indices