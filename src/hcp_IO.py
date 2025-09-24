
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

import pickle as pk
from scipy.io import loadmat
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform
from collections import OrderedDict


list_100unrelated = ['100307', '100408', '101107', '101309', '101915', '103111', '103414', '103818', '105014', '105115', '106016', '108828', '110411', '111312', '111716', '113619', '113922', '114419', '115320', '116524', '117122', '118528', '118730', '118932', '120111', '122317', '122620', '123117', '123925', '124422', '125525', '126325', '127630', '127933', '128127', '128632', '129028', '130013', '130316', '131217', '131722', '133019', '133928', '135225', '135932', '136833', '138534', '139637', '140925', '144832', '146432', '147737', '148335', '148840', '149337', '149539', '149741', '151223', '151526', '151627', '153025', '154734', '156637', '159340', '160123', '161731', '162733', '163129', '176542', '178950', '188347', '189450', '190031', '192540', '196750', '198451', '199655', '201111', '208226', '211417', '211720', '212318', '214423', '221319', '239944', '245333', '280739', '298051', '366446', '397760', '414229', '499566', '654754', '672756', '751348', '756055', '792564', '856766', '857263', '899885']
#######################################################################
#######################################################################
            #SCHAEFER300
#######################################################################
#######################################################################

def load_ts_schaefer300_df(path, transpose=False):
    fs = sorted(os.listdir(path))
    data = []
    for f in tqdm(fs):
        sub = f[:6]
        test = int(f.split('_')[1][-1])-1;
        ts_data = np.loadtxt(path+f,delimiter=',', skiprows=1)
        if transpose:
            ts_data = ts_data.T
        data.append((sub,test,ts_data))
    df = pd.DataFrame(data,columns=['subject','session','series'])
    print('TS are composed of')
    print(str(int(len(df['series'])/2)) + ' subjects')
    print(str(df['series'][0].shape[0]) + ' regions')
    print(str(df['series'][0].shape[1]) + ' time points')
    return df

def load_ts_schaefer300_dict(path, transpose=False):
    fs = sorted(os.listdir(path))
    ts_data = OrderedDict()
    for f in tqdm(fs):
        sub = f[:6]
        test = int(f.split('_')[1][-1])-1;
        if sub not in ts_data:
            ts_data[sub] = OrderedDict()
        ts_data[sub][test] = np.loadtxt(path+f,delimiter=',', skiprows=1)
        if transpose:
            ts_data[sub][test] = ts_data[sub][test].T
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
        sess = int(f[-1])
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
            path = data_dir+sub+f'/rfMRI_REST{test}_LR/Glasser360/TS_Glasser360S_gsr_bp.mat'
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
    list_s0 = [sub[0].flatten() for sub in matrices_dict.values()]
    list_s1 = [sub[1].flatten() for sub in matrices_dict.values()]
    n = len(list_s0)
    corr_mat_subjects = np.corrcoef(list_s0,list_s1)
    if path is not None:
        np.save(path+'.npy', corr_mat_subjects[n:,:n])
    else:
        return corr_mat_subjects[n:,:n]