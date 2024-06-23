import matplotlib.pyplot as plt
import os
import numpy as np
from config import config

# ---- plot attention maps ----
def plot_attention_maps(input_data, attn_maps, idx=0):
    if not os.path.exists('results'):
        os.makedirs('results')

    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads*fig_size, num_layers*fig_size), dpi=300)
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    avg_attn = np.zeros((seq_len, seq_len))
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(attn_maps[row][column], origin='lower')
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist(), fontsize=4)
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist(), fontsize=4)
            ax[row][column].set_title(f"Layer {row+1}, Head {column+1}")
            avg_attn += attn_maps[row][column]
    fig.subplots_adjust(hspace=0.5)
    plt.savefig('results/attn_maps.png')
    plt.show()
    plt.close()
    
    avg_attn /= num_heads*num_layers
    plt.figure(dpi=300)
    plt.imshow(avg_attn, origin='lower')
    plt.xticks(list(range(seq_len)), fontsize=4)
    plt.yticks(list(range(seq_len)), fontsize=4)
    plt.colorbar()
    # ax[row][column].set_xticks(list(range(seq_len)))
    # ax[row][column].set_xticklabels(input_data.tolist())
    plt.title('average attention map')
    plt.savefig('results/avg_attn_map.png')
    plt.show()
    plt.close()
    return avg_attn
# --------------------------------

def plot_example_batch(dataloader):
    # create foler to save images
    if not os.path.exists('results'):
        os.makedirs('results')
    # get 1 batch of data 
    ids, x, y = next(iter(dataloader))
    cmap='rainbow'
    fig, axs = plt.subplots(1,2)
    axs[0].title.set_text('Input')
    axs[1].title.set_text('Target')
    fig.colorbar(axs[0].imshow(x[:config.batch_size,:].numpy(),cmap=cmap, aspect='auto'), ax=axs[0], shrink=0.6, orientation='horizontal')
    fig.colorbar(axs[1].imshow(y[:config.batch_size,:].numpy(),cmap=cmap, aspect='auto'), ax=axs[1], shrink=0.6, orientation='horizontal')
    plt.savefig('results/example_batch.png')
    plt.show()
    plt.close()

def plot_predictions(input, target, pred):
    if not os.path.exists('results'):
        os.makedirs('results')
    cmap='rainbow'
    fig, axs = plt.subplots(1,3)
    axs[0].title.set_text('Input')
    axs[1].title.set_text('Target')
    axs[2].title.set_text('Prediction')
    fig.colorbar(axs[0].imshow(input,cmap=cmap, aspect='auto'), ax=axs[0], shrink=0.6, orientation='horizontal')
    fig.colorbar(axs[1].imshow(target,cmap=cmap, aspect='auto'), ax=axs[1], shrink=0.6, orientation='horizontal')
    fig.colorbar(axs[2].imshow(pred,cmap=cmap, aspect='auto'), ax=axs[2], shrink=0.6, orientation='horizontal')
    plt.savefig('results/predictions.png')
    plt.show()
    plt.close()
