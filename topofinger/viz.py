import numpy as np
from itertools import combinations
import random
import matplotlib.pyplot as plt

import pandas as pd

from pingouin import compute_effsize

import scipy
import seaborn as sns
from scipy.spatial.distance import squareform
from tqdm.notebook import tqdm

from .utils import threshold_matrix
from .fingerprinting import compute_idiff, compute_effect_size, compute_success_rate


def plot_identifiability(sim_matrix, ax=None, title='title', kind='corr', vmin=None, vmax=None, cmap=None):
    if ax is None:
        ax = plt.gca()

    #check and remove Nans
    #check and remove NaNs columns
    nan_rows = np.isnan(sim_matrix).all(axis=1)
    nan_cols = np.isnan(sim_matrix).all(axis=0)
    nans = nan_cols | nan_rows
    sim_matrix = sim_matrix[~nans][:, ~nans]

    # Compute metrics for identifiability
    #idiff, i_self, i_others = compute_idiff(sim_matrix, kind=kind)
    effect_size = compute_effect_size(sim_matrix, kind=kind)
    success_rate1, success_rate2 = compute_success_rate(sim_matrix, kind=kind)
    
    # Create the plot
    #fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_title(title, fontsize=18)
    if kind == 'corr':
        im = ax.imshow(sim_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        #im = ax.imshow(sim_matrix,cmap='Oranges')
    elif kind == 'dist':
        im = ax.imshow(sim_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    #ax.axis('off')
    #plt.xlabel = 'session 1'
    #ax.set_xlabel('Subjects, session 1', fontsize=14)
    #ax.set_ylabel('Subjects, session 2', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_ylabel =('session 2')
    #ax.text(0.5, -0.35, f'Idiff = {round(idiff, 2)} (Is={round(i_self, 2)}, Io={round(i_others, 2)})\nEffect Size = {round(effect_size, 2)}\n\nSuccess Rate s1 = {round(success_rate1, 2)}%\nSuccess Rate s2 = {round(success_rate2, 2)}%', transform=ax.transAxes, ha='center', fontsize=14)
    #ax.text(0.5, -0.35, f'Effect Size = {round(effect_size, 2)}\n\nSuccess Rate s1 = {round(success_rate1, 2)}%\nSuccess Rate s2 = {round(success_rate2, 2)}%', transform=ax.transAxes, ha='center', fontsize=14)
    ax.text(0.5, -0.1, f'Effect Size = {round(effect_size, 2)}', transform=ax.transAxes, ha='center', fontsize=14)
    ax.text(0.5, -0.2, f'Success Rate = {(success_rate1+success_rate2)/2:.1f}', transform=ax.transAxes, ha='center', fontsize=14)
    
    

    #ax.text(0.5, -0.2, f'Success Rate = {round((success_rate1+success_rate2)/2, 2)}', transform=ax.transAxes, ha='center', fontsize=14)
    #ax.text(0.5, -0.2,f'\nSuccess Rate s1 = {round(success_rate1, 2)}%\nSuccess Rate s2 = {round(success_rate2, 2)}%', transform=ax.transAxes, ha='center', fontsize=14)

    return ax


def comparison_distances(data_dict, ncomparison=30, between=False):
    
    #do comparison within session or between different sessions
    if between == True:
        sess = 1
    else:
        sess = 0
        
    #initialize dicts for storing distances
    insub_dists = {}
    outsub_dists = {}
    
    #loop over subjects
    for sub in data_dict:
        insub_dists[sub] = [] #list of distances of subj with himself
        outsub_dists[sub] = [] #list of differences of subj with other subs
        
        #define maximum number of comparison based on number of windows
        N = len(data_dict[sub][0].keys())
        N = np.min([N,ncomparison])
        
        #list all possible combination of two different windows
        list_combo = list(combinations(data_dict[sub][0].keys(), 2))
        random.shuffle(list_combo)
        
        for w1,w2 in list_combo[:N]:
            F = data_dict[sub][0][w1]
            G = data_dict[sub][sess][w2]
            if scipy.sparse.issparse(F):
                F = np.array(F.todense())
                G = np.array(G.todense())
            d = np.linalg.norm(F-G) / np.linalg.norm(F+G)
            insub_dists[sub].append(d)
        
        count = len(insub_dists[sub])
        while count > 0:
            count -= 1
            subs = list(data_dict.keys())
            subs.remove(sub)
            new_sub = np.random.choice(subs)
            i,j = np.random.choice(range(len(data_dict[sub][0].keys())), 2)
            F = data_dict[sub][0][i]
            G = data_dict[new_sub][sess][j]
            if scipy.sparse.issparse(F):
                F = np.array(F.todense())
                G = np.array(G.todense())
            d = np.linalg.norm(F-G) / np.linalg.norm(F+G)
            outsub_dists[sub].append(d)
            
    return insub_dists, outsub_dists


#OLD Threshold
'''def threshold_adj_matrix(matrix, perc_to_keep=0.01):
    matrix = (matrix+matrix.T)/2
    vector = squareform(matrix,checks=False)
    vector_sorted = np.sort(vector)[::-1]
    min_to_keep = int(len(vector_sorted)*perc_to_keep)-1
    thresh = vector_sorted[min_to_keep]
    
    vector[vector < thresh] = 0
    
    matrix = squareform(vector)
    return matrix
'''



def comparison_distances_whole(data_dict, perc_to_keep=25):
    #initialize dicts for storing distances
    insub_dists = {}
    outsub_dists = {}
    #loop over subjects
    for sub in tqdm(data_dict):
        insub_dists[sub] = [] #list of distances of subj with himself
        outsub_dists[sub] = [] #list of differences of subj with other subs
        
        #insub
        F = data_dict[sub][0]
        G = data_dict[sub][1]
        
        #threshold
        F = threshold_matrix(F, perc_to_keep=perc_to_keep)
        G = threshold_matrix(G,perc_to_keep=perc_to_keep)
        
        if scipy.sparse.issparse(F):
            F = np.array(F.todense())
            G = np.array(G.todense())
        d = np.linalg.norm(F-G) / np.linalg.norm(F+G)
        insub_dists[sub].append(d)
        
        #outsub
        
        for sub2 in data_dict:
            if sub2!=sub:
                G = data_dict[sub2][1]
                
                #threshold
                G = threshold_matrix(G,perc_to_keep=perc_to_keep)
                if scipy.sparse.issparse(G):
                    G = np.array(G.todense())
                d = np.linalg.norm(F-G) / np.linalg.norm(F+G)
                outsub_dists[sub].append(d)            
    return insub_dists, outsub_dists

    
    

def comparison_distances_thr(data_dict, ncomparison=30, between=False, perc_to_keep=25):
    
    #do comparison within session or between different sessions
    if between == True:
        sess = 1
    else:
        sess = 0
        
    #initialize dicts for storing distances
    insub_dists = {}
    outsub_dists = {}
    
    #loop over subjects
    for sub in data_dict:
        insub_dists[sub] = [] #list of distances of subj with himself
        outsub_dists[sub] = [] #list of differences of subj with other subs
        
        #define maximum number of comparison based on number of windows
        N = len(data_dict[sub][0].keys())
        N = np.min([N,ncomparison])
        
        #list all possible combination of two different windows
        list_combo = list(combinations(data_dict[sub][0].keys(), 2))
        random.shuffle(list_combo)
        
        for w1,w2 in list_combo[:N]:
            F = data_dict[sub][0][w1]
            G = data_dict[sub][sess][w2]
            
            ## set self loop to zero
            np.fill_diagonal(F,0)
            np.fill_diagonal(G,0)
            
            #threshold
            F = threshold_matrix(F, perc_to_keep=perc_to_keep)
            G = threshold_matrix(G,perc_to_keep=perc_to_keep)
            
            if scipy.sparse.issparse(F):
                F = np.array(F.todense())
                G = np.array(G.todense())
            d = np.linalg.norm(F-G) / np.linalg.norm(F+G)
            insub_dists[sub].append(d)
        
        count = len(insub_dists[sub])
        while count > 0:
            count -= 1
            subs = list(data_dict.keys())
            subs.remove(sub)
            new_sub = np.random.choice(subs)
            i,j = np.random.choice(range(len(data_dict[sub][0].keys())), 2)
            F = data_dict[sub][0][i]
            G = data_dict[new_sub][sess][j]
            
            ## set self loop to zero
            np.fill_diagonal(F,0)
            np.fill_diagonal(G,0)
            
            #threshold
            F = threshold_matrix(F,perc_to_keep=perc_to_keep)
            G = threshold_matrix(G,perc_to_keep=perc_to_keep)
            
            if scipy.sparse.issparse(F):
                F = np.array(F.todense())
                G = np.array(G.todense())
            d = np.linalg.norm(F-G) / np.linalg.norm(F+G)
            outsub_dists[sub].append(d)
            
    return insub_dists, outsub_dists



def visualize_plot(insub_dists,outsub_dists):
    
    plt.figure(figsize=(8,4))
    plt.style.use('seaborn-ticks')

    xs = np.array(list(map(lambda x: np.mean(insub_dists[x]), insub_dists.keys())))
    ys = np.array(list(map(lambda x: np.mean(outsub_dists[x]), insub_dists.keys())))
    plt.plot(xs, ys, 'o',alpha=.6)
    plt.plot(np.linspace(0,1,10), np.linspace(0,1,10),'--k')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('same-subject dist',fontsize=20)
    plt.ylabel('different-subjects dist',fontsize=20)
    plt.show()   
    
def intraclass_corr(x,y,k=2):
    return (np.std(y)-np.std(x))/(np.std(y) + (k-1)*np.std(x))
   
def visualize_bplot(insub_dists,outsub_dists,title=''):
    idx = []
    rho = []
    df = pd.DataFrame()
    xs = 2*np.array(list(map(lambda x: np.mean(insub_dists[x]), insub_dists.keys())))
    ys = 2*np.array(list(map(lambda x: np.mean(outsub_dists[x]), insub_dists.keys())))
    
    pop1, pop2 =  xs, ys
    rho.append(intraclass_corr(xs,ys))
    Ns = np.min([len(pop1),len(pop2)])
    choice_kwargs = dict(size=Ns, replace=False)
    g1 = np.random.choice(pop1, **choice_kwargs)
    g2 = np.random.choice(pop2, **choice_kwargs)
    inkey = 'in '
    df[inkey] = g1
    outkey = 'across '
    df[outkey] = g2
    idx.append((inkey,outkey))
    
    fig = plt.figure(figsize=(6,2))
    positions = [0,1]
    medianprops = dict(linestyle='-', linewidth=2.5, color='slategrey')
    whiskerprops = dict(linestyle='-', color='slategrey')
    bplot = plt.boxplot(df.values, tick_labels=df.columns, patch_artist=True,showbox=False,
                        showcaps=False,medianprops=medianprops,whiskerprops=whiskerprops, positions=positions); 
    plt.ylabel('avg distance', fontsize=20)
    effect_size = np.round(compute_effsize(df[df.columns[0]], df[df.columns[1]]),3)
    plt.title(str(title)+' E.S. = ' + str(effect_size), fontsize=20)
    plt.xticks(rotation=30, fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    
    return effect_size

def visualize_bplot_new(insub_dists,outsub_dists,title=''):
    idx = []
    rho = []
    df = pd.DataFrame()
    xs = 2*np.array(list(map(lambda x: np.mean(insub_dists[x]), insub_dists.keys())))
    ys = 2*np.array(list(map(lambda x: np.mean(outsub_dists[x]), insub_dists.keys())))
    
    inkey = 'intra-subject '
    df[inkey] = xs
    outkey = 'inter-subject '
    df[outkey] = ys
    idx.append((inkey,outkey))
    
    fig = plt.figure(figsize=(5,4))
    positions = [0,1]
    medianprops = dict(linestyle='-', linewidth=1.5, color='black')
    whiskerprops = dict(linestyle='-', color='black')
    
    #bplot = sns.boxplot(df.values, labels=df.columns)
    bplot = plt.boxplot(df.values, tick_labels=df.columns, patch_artist=True, showbox=True,
                        showcaps=True,medianprops=medianprops,whiskerprops=whiskerprops, positions=positions);
    for patch in bplot['boxes']:
        patch.set_facecolor('#d88f8f')
    plt.ylabel('Distances', fontsize=12)
    effect_size = np.round(compute_effsize(ys, xs),3)
    plt.title('Effect Size = ' + str(effect_size), fontsize=15)
    #plt.title(str(title)+' E.S. = ' + str(effect_size), fontsize=20)
    plt.xticks(rotation=30, fontsize=12)
    plt.yticks(fontsize=10)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f'/data/shared/spoetto/net-sci figure/eff sizes/{title}.png', transparent=True)    

    plt.show()


    import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_connectivity_matrix_functional_subnetworks(connectivity_matrix, labels, subnet_order, patch_colors, subnet_ranges, ax=None, vmin=None, vmax=None):
    """
    Plots a functional connectivity matrix in a given axis (ax) and creates small
    patches outside the main plot (as additional axes) for labeling. This allows
    integration into complex subplot layouts such as mosaic subplots.

    Args:
        connectivity_matrix (numpy.ndarray): The connectivity matrix (square, N x N).
        labels (list): List of labels for the patches (e.g., ['SM', 'DA', 'VA', ...]).
        patch_colors (list): List of colors for the patches.
        subnet_ranges (list of tuples): Start/end indices for each label range.
        ax (matplotlib.axes._axes.Axes, optional): Axis to plot into. Creates a
                                                   new figure and axis if None.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    connectivity_matrix = connectivity_matrix[subnet_order][:, subnet_order]

    if vmin!=None:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(connectivity_matrix, cmap='coolwarm', interpolation='nearest', norm=norm)
    else:
        im = ax.imshow(connectivity_matrix, cmap='coolwarm', interpolation='nearest')

    ax.set_xlim(-0.5, len(connectivity_matrix) - 0.5)
    ax.set_ylim(len(connectivity_matrix) - 0.5, -0.5)

    # Position boxes relative to the main axis
    bb = ax.get_position()
    box_height = 0.1 * bb.height
    box_width = 0.1 * bb.width

    ax_x_patches = fig.add_axes([bb.x0, bb.y0 + bb.height, bb.width, box_height])
    ax_y_patches = fig.add_axes([bb.x0 - box_width, bb.y0, box_width, bb.height])

    ax_x_patches.set_xlim(0, len(connectivity_matrix))
    ax_x_patches.set_ylim(0, 1)
    ax_y_patches.set_xlim(0, 1)
    ax_y_patches.set_ylim(0, len(connectivity_matrix))

    cax = fig.add_axes([bb.x0 + bb.width + 0.01, bb.y0, 0.01, bb.height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=6)

    for i, label in enumerate(labels):
        color = patch_colors[i]
        start, end = subnet_ranges[i]
        width = end - start
        ax_x_patches.set_in_layout(False)
        ax_y_patches.set_in_layout(False)
        rect_x = patches.Rectangle((start, 0), width, 0.5, facecolor=color)
        ax_x_patches.add_patch(rect_x)
        ax_x_patches.text(start + width / 2, 0.25, label, ha='center', va='center', fontsize=6)

        height = end - start
        rect_y = patches.Rectangle((0.5, start), 0.5, height, facecolor=color)
        ax_y_patches.add_patch(rect_y)
        ax_y_patches.text(0.75, start + height / 2, label, ha='center', va='center', rotation=90, fontsize=6)

    for patch_ax in [ax_x_patches, ax_y_patches]:
        patch_ax.set_xticks([])
        patch_ax.set_yticks([])
        patch_ax.set_frame_on(False)

        for spine in patch_ax.spines.values():
            spine.set_visible(False)

    ax_y_patches.invert_yaxis()
            
    #cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #cbar.ax.tick_params(labelsize=6)
    ax.set_xticks([])
    ax.set_xticklabels([])
    return ax