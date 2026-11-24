
import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd

import pickle as pk
from scipy.io import loadmat
from scipy.stats import pearsonr, zscore
from scipy.spatial.distance import squareform
from collections import OrderedDict



list_100unrelated = ['100307', '100408', '101107', '101309', '101915', '103111', '103414', '103818', '105014', '105115', '106016', '108828', '110411', '111312', '111716', '113619', '113922', '114419', '115320', '116524', '117122', '118528', '118730', '118932', '120111', '122317', '122620', '123117', '123925', '124422', '125525', '126325', '127630', '127933', '128127', '128632', '129028', '130013', '130316', '131217', '131722', '133019', '133928', '135225', '135932', '136833', '138534', '139637', '140925', '144832', '146432', '147737', '148335', '148840', '149337', '149539', '149741', '151223', '151526', '151627', '153025', '154734', '156637', '159340', '160123', '161731', '162733', '163129', '176542', '178950', '188347', '189450', '190031', '192540', '196750', '198451', '199655', '201111', '208226', '211417', '211720', '212318', '214423', '221319', '239944', '245333', '280739', '298051', '366446', '397760', '414229', '499566', '654754', '672756', '751348', '756055', '792564', '856766', '857263', '899885']
#######################################################################
#######################################################################
            #SCHAEFER300
#######################################################################
#######################################################################

def load_ts_schaefer300_df(path):
    fs = sorted(os.listdir(path))
    data = []
    for f in tqdm(fs):
        sub = f[:6]
        test = int(f.split('_')[1][-1])-1;
        ts_data = np.loadtxt(path+f,delimiter=',', skiprows=1).T
        data.append((sub,test,ts_data))
    df = pd.DataFrame(data,columns=['subject','session','series'])
    print('TS are composed of')
    print(str(int(len(df['series'])/2)) + ' subjects')
    print(str(df['series'][0].shape[0]) + ' regions')
    print(str(df['series'][0].shape[1]) + ' time points')
    return df

def load_ts_1200subs(path):
    fs=sorted(os.listdir(path))
    data = []
    for f in tqdm(fs):
        sub = f[:6]
        test = int(f.split('_')[2][-1])-1;
        ts_data = np.load(path+f)['timeseries']
        data.append((sub,test,ts_data))
    df = pd.DataFrame(data,columns=['subject','session','series'])
    print('TS are composed of')
    print(str(int(len(df['series'])/2)) + ' subjects')
    print(str(df['series'][0].shape[0]) + ' regions')
    print(str(df['series'][0].shape[1]) + ' time points')
    return df


def load_ts_schaefer300_dict(path, transpose=False):
    """Load Schaefer300 time series from a folder of .csv or .npz files.

    The session is read from 'REST<n>' in the file name (e.g. 100307_REST1_LR.csv or
    100206_rfMRI_REST1_LR_Schaefer300+Tian_NoGSR_LowPassOnly.npz).
    Returns data[subject][session] = array [TIMESTEPS x ROIs]; with transpose=True, [ROIs x TIMESTEPS].
    NaNs are set to 0.
    """
    fs = sorted(f for f in os.listdir(path) if f.endswith(('.csv', '.npz')))
    ts_data = OrderedDict()
    for f in tqdm(fs):
        sub = f[:6]
        test = int(re.search(r'REST(\d)', f).group(1)) - 1
        if sub not in ts_data:
            ts_data[sub] = OrderedDict()
        if f.endswith('.npz'):
            ts = np.load(path + f)['timeseries'].T  # stored as [ROIs x TIMESTEPS]
        else:
            ts = np.loadtxt(path + f, delimiter=',', skiprows=1)
        ts = np.nan_to_num(ts)
        ts_data[sub][test] = ts.T if transpose else ts
    return ts_data

def load_fc_schaefer300(path):
    with open(path, 'rb') as f:
        fc = pk.load(f)
    return fc

def load_scaffold_schaefer300(path):
    fs = sorted(os.listdir(path))
    sc = OrderedDict()
    for f in fs:
        sub = f[:6]
        if sub in sc:
            pass
        else:
            sc[sub] = OrderedDict()
        sess = int(f[-5])
        path_file = path+f
        data = np.load(path_file)
        sc[sub][sess] = data
    return sc    

def load_representatives_schaefer300(path):
    fs = sorted(os.listdir(path))
    representatives = OrderedDict()
    for f in fs:
        sub = f[3:9]
        rep0 = np.load(path+f, allow_pickle=True)[0]
        rep1 = np.load(path+f, allow_pickle=True)[1]
        representatives[sub] = (rep0,rep1)
    return representatives

#######################################################################
#######################################################################
            #GLASSER379
#######################################################################
#######################################################################

def load_ts_glasser379_dict(data_dir):
    #load data and create a dictionary: data[subject][test]= matrix[TIMESTEPS x ROIs]
    ts_data = OrderedDict()
    fs = os.listdir(data_dir)
    for sub in tqdm(fs):
        ts_data[sub] = OrderedDict()
        for test in [1,2]:
            #path = data_dir+sub+f'/rfMRI_REST{test}_LR/Glasser360/TS_Glasser360S_gsr_bp.mat'
            path = data_dir+sub+f'/rfMRI_REST{test}_LR/Glasser360/TS_Glasser360S_bp.mat'
            data = loadmat(path)
            ts_data[sub][test-1] = data['TS'].T
    return ts_data

#######################################################################
#######################################################################
            #SCHAEFER100
#######################################################################
#######################################################################

def load_ts_schaefer100_dict(data_dir):
    #load data and create a dictionary: data[subject][test]= matrix[TIMESTEPS x ROIs]
    ts_data = OrderedDict()
    fs = os.listdir(data_dir)
    for sub in tqdm(fs):
        ts_data[sub] = OrderedDict()
        for test in [1,2]:
            path = data_dir+sub+f'/rfMRI_REST{test}_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
            data = loadmat(path)
            ts_data[sub][test-1] = data['TS'].T
    return ts_data


#######################################################################
#######################################################################
            #AAL89
#######################################################################
#######################################################################
def load_ts_aal89_dict(data_dir):
    #load data and create a dictionary: data[subject][test]= matrix[TIMESTEPS x ROIs]
    ts_data = OrderedDict()
    fs = os.listdir(data_dir)
    for f in tqdm(fs):
        sub = f.split('_')[1]
        test = int(f.split('_')[3][-1]);
        if sub not in ts_data:
            ts_data[sub] = OrderedDict()
        ts_data[sub][test] = np.loadtxt(data_dir+f)
    return ts_data

#######################################################################
#######################################################################
            #SHEN268
#######################################################################
#######################################################################
def load_ts_shen268_dict(data_dir):
    #load data and create a dictionary: data[subject][test]= matrix[TIMESTEPS x ROIs]
    ts_data = OrderedDict()
    fs = os.listdir(data_dir)
    for sub in tqdm(fs):
        ts_data[sub] = OrderedDict()
        ts_data[sub][0] = np.load(data_dir+sub+f'/{sub}_REST1_LR.npy')
        ts_data[sub][1] = np.load(data_dir+sub+f'/{sub}_REST2_LR.npy')
    return ts_data

#######################################################################
#######################################################################
            #LOAD
#######################################################################
#######################################################################
def load_scaffold(path):
    fs = sorted(os.listdir(path))
    sc = OrderedDict()
    for f in fs:
        sub = f[:6]
        if sub in sc:
            pass
        else:
            sc[sub] = OrderedDict()
        sess = int(f[-1])
        path_file = path+f
        data = np.load(path_file)
        sc[sub][sess] = data
    return sc 

def load_behavioural_scores(path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(path)
    df['Subject'] = df['Subject'].astype(str)
    df = df[df['Subject'].isin(list_100unrelated)]
    df.set_index('Subject', inplace=True)
    #filter only numerical values
    df = df.select_dtypes(include='number')
    #remove columns with nans
    columns_with_nans = df.columns[df.isna().any()]
    df = df.drop(columns=columns_with_nans)
    # Calculate the variance of each column
    variances = df.var()
    # Filter out columns with zero variance
    df = df.loc[:, variances != 0]
    #remove the first columns about study completion
    df = df.iloc[:, 8:]
    return df

    
#######################################################################
#######################################################################
            #SAVE
#######################################################################
#######################################################################

def save_whole_correlations_dict(ts_data, path):
    #Compute correlation matrix for the whole time series: whole_corrs[subject][test] = matrix(ROIs x ROIs)
    whole_corrs = OrderedDict()
    for sub in ts_data:
        whole_corrs[sub] = OrderedDict()
        for test in ts_data[sub]:
            whole_corrs[sub][test] = np.corrcoef(ts_data[sub][test].T)
    if not os.path.exists(path+'fc/'):
        os.makedirs(path+'fc/')
    if not os.path.exists(path+'fc/whole/'):
        os.makedirs(path+'fc/whole/')
    with open(path+'fc/whole/whole_corrs.pck','wb') as f:
        pk.dump(whole_corrs,f)
        
def save_distances_subjects(matrices_dict, path):
    list_s0 = [sub[0] for sub in matrices_dict.values()]
    list_s1 = [sub[1] for sub in matrices_dict.values()]
    n = len(list_s0)
    dist_mat_subjects = np.zeros((n,n))
    for i in tqdm(range(n)):
        for j in range(n):
            F = list_s0[i]
            G = list_s1[j]
            d = np.linalg.norm(F-G) / np.linalg.norm(F+G)
            dist_mat_subjects[i,j] = d
    np.save(path+'.npy', dist_mat_subjects)
    
def save_correlations_subjects(matrices_dict, path):
    list_s0 = [sub[0] for sub in matrices_dict.values()]
    list_s1 = [sub[1] for sub in matrices_dict.values()]
    n = len(list_s0)
    corr_mat_subjects = np.zeros((n,n))
    for i in tqdm(range(n)):
        for j in range(n):
            F = np.nan_to_num(list_s0[i])
            G = np.nan_to_num(list_s1[j])
            c = pearsonr(F.flatten(),G.flatten())[0]
            corr_mat_subjects[i,j] = c
    np.save(path+'.npy', corr_mat_subjects)
    
def save_correlations_subjects_fast(matrices_dict, path=None):
    #list_s0 = [sub[0].flatten() for sub in matrices_dict.values()]
    #list_s1 = [sub[1].flatten() for sub in matrices_dict.values()]
    list_s0 = [np.nan_to_num(zscore(sub[0], axis=None),0).flatten() for sub in matrices_dict.values()]
    list_s1 = [np.nan_to_num(zscore(sub[1], axis=None),0).flatten() for sub in matrices_dict.values()]
    
    n = len(list_s0)
    corr_mat_subjects = np.corrcoef(list_s0,list_s1)
    if path is None:
        return corr_mat_subjects[n:,:n]
    np.save(path+'.npy', corr_mat_subjects[n:,:n])

def load_ts_custom(data_dir, lr='LR', atlas='Schaefer300', gsr='NoGSR', bandpass='LowPassOnly'):
    ts_data = OrderedDict()
    fs = sorted(os.listdir(data_dir))

    # Extract unique subjects from filenames
    subjects = sorted(set(f.split('_')[0] for f in fs))

    for sub in tqdm(subjects):
        # Check if both sessions exist before loading
        patterns = [
            f"{sub}_rfMRI_REST{n}_{lr}_{atlas}+Tian_{gsr}_{bandpass}.npz"
            for n in [1, 2]
        ]
        if not all(os.path.exists(os.path.join(data_dir, p)) for p in patterns):
            continue

        ts_data[sub] = OrderedDict()
        for idx, pattern in enumerate(patterns):
            data = np.load(os.path.join(data_dir, pattern))
            ts_data[sub][idx] = data['timeseries'].T
    return ts_data


def crop_regions(d, n):
    """Return a copy of a nested dict d[sub][sess] = square matrix, keeping only the first n regions."""
    return OrderedDict((sub, OrderedDict((ses, m[:n, :n]) for ses, m in sess.items())) for sub, sess in d.items())


def load_scaffold_cortical(variant_dir, n=300, verbose=True, loader=None):
    """Load whole-scan frequency scaffolds restricted to the n cortical regions of a HCP-1200 variant folder.

    Uses scaffolds/whole_cortex/freq (scaffolds recomputed on the cortical regions only) if it exists,
    otherwise crops the 316-region scaffolds in scaffolds/whole/freq (approximation: cycles were
    computed with the subcortical nodes present).
    Also used for Glasser (n=360, loader=load_scaffold, whose files have no extension).
    """
    loader = loader or load_scaffold_schaefer300
    exact = os.path.join(variant_dir, 'scaffolds', 'whole_cortex', 'freq') + '/'
    full = os.path.join(variant_dir, 'scaffolds', 'whole', 'freq') + '/'
    if os.path.isdir(exact) and len(os.listdir(exact)) >= len(os.listdir(full)):  # only when complete
        if verbose:
            print(f'  scaffolds: recomputed on cortical regions ({exact})')
        return loader(exact)
    if verbose:
        print(f'  scaffolds: CROPPED to the first {n} regions ({full})')
    return crop_regions(loader(full), n)


def load_fc_cortical(variant_dir, n=300):
    """Load whole-scan FC (fc/whole/whole_corrs.pck) restricted to the n cortical regions (exact for Pearson FC).

    NaNs (regions with a flat time series, e.g. Glasser subjects 856766 and 101107) are set to 0.
    """
    with open(os.path.join(variant_dir, 'fc', 'whole', 'whole_corrs.pck'), 'rb') as f:
        fc = pk.load(f)
    fc = crop_regions(fc, n)
    for sess in fc.values():
        for k in sess:
            sess[k] = np.nan_to_num(sess[k])
    return fc
