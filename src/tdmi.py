'''
Utils for computing integrated information decopomposition of signals
based on the gaussian copula
'''


import gcmi

import numpy as np

from itertools import combinations

from tqdm.notebook import tqdm

from hoi.metrics import atoms_phiID, RedundancyphiID, psi_synergy

from scipy.spatial.distance import squareform

def red_TDMI(ts1,ts2,delay=1):
    x_past = ts1[:-delay]
    x_present = ts1[delay:]
    y_past = ts2[:-delay]
    y_present = ts2[delay:]
    
    past = np.vstack((x_past,y_past))
    present = np.vstack((x_present,y_present))
    
    list_redundancy = []
    list_redundancy.append(gcmi.gcmi_cc(x_past,x_present))
    list_redundancy.append(gcmi.gcmi_cc(x_past,y_present))
    list_redundancy.append(gcmi.gcmi_cc(y_past,x_present))
    list_redundancy.append(gcmi.gcmi_cc(y_past,y_present))
    
    redundancy = min(list_redundancy)
    return redundancy


def syn_TDMI(ts1,ts2,delay=1):
    x_past = ts1[:-delay]
    x_present = ts1[delay:]
    y_past = ts2[:-delay]
    y_present = ts2[delay:]
    
    past = np.vstack((x_past,y_past))
    present = np.vstack((x_present,y_present))
    
    mi_past_present = gcmi.gcmi_cc(past,present)
    mi_xy_present = max((gcmi.gcmi_cc(x_past,present),gcmi.gcmi_cc(y_past,present)))

    synergy = mi_past_present-mi_xy_present
    return synergy

def synR_TDMI(ts1,ts2,delay=1):
    x_past = ts1[:-delay]
    x_present = ts1[delay:]
    y_past = ts2[:-delay]
    y_present = ts2[delay:]
    
    past = np.vstack((x_past,y_past))
    present = np.vstack((x_present,y_present))
    
    mi_past_present = gcmi.gcmi_cc(past,present)
    mi_xy_present = max((gcmi.gcmi_cc(x_past,present),gcmi.gcmi_cc(y_past,present)))

    synergy = mi_past_present-mi_xy_present
    redundancy = red_TDMI(ts1,ts2,delay=delay)
    synergy = synergy + redundancy
    return synergy


def non_unique_TDMI(ts1,ts2,delay=1):
    x_past = ts1[:-delay]
    x_present = ts1[delay:]
    y_past = ts2[:-delay]
    y_present = ts2[delay:]
    
    past = np.vstack((x_past,y_past))
    present = np.vstack((x_present,y_present))
    
    list_redundancy = []
    list_redundancy.append(gcmi.gcmi_cc(x_past,x_present))
    list_redundancy.append(gcmi.gcmi_cc(x_past,y_present))
    list_redundancy.append(gcmi.gcmi_cc(y_past,x_present))
    list_redundancy.append(gcmi.gcmi_cc(y_past,y_present))
    redundancy = min(list_redundancy)
    
    mi_past_present = gcmi.gcmi_cc(past,present)
    mi_xy_present = max((gcmi.gcmi_cc(x_past,present),gcmi.gcmi_cc(y_past,present)))

    synergy = mi_past_present-mi_xy_present
    
    return mi_past_present-synergy-redundancy


def syn_matrix(data, delay=1, kind='tdmi'):
    """
    Calculate the synergy matrix between pair of timeseries.

    Parameters:
    - data: numpy.ndarray
        The input data array of shape (N, T), where N is the number of timeseries and T is their lenght
    - delay: int, optional
        The time delay for calculating time delayed mutual information. Default is 1.

    Returns:
    - syn_matrix: numpy.ndarray
        The synergy matrix of shape (N, N), where N is the number of timeseries.
    """

    N = data.shape[0]
    syn_matrix = np.zeros((N,N))
    if kind == 'tdmi':
        for i1,i2 in combinations(range(N),2):
            syn = syn_TDMI(data[i1],data[i2],delay=delay)
            syn_matrix[i1,i2] = syn
        syn_matrix = syn_matrix + syn_matrix.T
    elif kind == 'atoms_phi':
        model = atoms_phiID(data)
        results = model.fit(minsize = 2, maxsize=2)
        syn_matrix = squareform(results[:,0])
    elif kind == 'psi':
        model = psi_synergy(data.T)
        results = model.fit(minsize = 2, maxsize=2)
        syn_matrix = squareform(results[:,0])  
    elif kind == 'phiR':
        model = atoms_phiID(data.T)
        results = model.fit(minsize=2, maxsize=2, atoms=['sts','u1ts','u2ts','stu2','stu1','rts','str','u1tu2','u2tu1'])
        syn_matrix = squareform(results[:,0])
    else:
        raise ValueError("kind must be 'tdmi' or 'atoms_phi' or 'phiR'")
    return syn_matrix


def red_matrix(data, delay=1, kind='tdmi'):
    """
    Calculates the redundancy matrix between pair of timeseries.

    Parameters:
    - data: numpy.ndarray
        The input data array of shape (N, T), where N is the number of timeseries and T is their lenght
    - delay: int, optional
        The time delay for calculating TDMI. Default is 1.

    Returns:
    - red_matrix: numpy.ndarray
        The synergy matrix of shape (N, N), where N is the number of timeseries.
    """
    
    N = data.shape[0]
    red_matrix = np.zeros((N,N))

    if kind == 'tdmi':
        
        for i1,i2 in combinations(range(N),2):
            red = red_TDMI(data[i1],data[i2],delay=delay)
            red_matrix[i1,i2] = red
        
        red_matrix = red_matrix+red_matrix.T
    
    elif kind == 'atoms_phi':
        model = RedundancyphiID(data.T)
        results = model.fit(minsize=2, maxsize=2, method='gauss')
        red_matrix = squareform(results[:,0])
    elif kind == 'psi':
        model = psi_redundancy(data.T)
        results = model.fit(minsize = 2, maxsize=2, method ='gauss')
        red_matrix = squareform(results[:,0])
    else:
        raise ValueError("kind must be 'tdmi' or 'atoms_phi'")
    
    return red_matrix


def non_unique_info_matrix(data,delay=1):
    N = data.shape[0]
    non_unique_info_matrix = np.zeros((N,N))
    
    for i1,i2 in combinations(range(N),2):
        non_unique = non_unique_TDMI(data[i1],data[i2],delay=delay)
        non_unique_info_matrix[i1,i2] = non_unique
    non_unique_info_matrix = non_unique_info_matrix + non_unique_info_matrix.T
    return non_unique_info_matrix


def sequential_syn_matrices(ts,width,shift):
    N, T = ts.shape; #ts lenght and n channels
    no_win = int((T - width)/shift) #number of windows
    syns = np.zeros((N, N, no_win)); #init correlation matrix
    for t in tqdm(range(no_win)):
        synergy_mat = np.zeros((N,N))
        for i1,i2 in combinations(range(N),2):
            syn = syn_TDMI(ts[i1,t*shift: t*shift+width],ts[i2,t*shift: t*shift+width])
            synergy_mat[i1,i2] = syn
            syns[:,:,t] = synergy_mat + synergy_mat.T

    return syns

def sequential_red_matrices(ts,width,shift):
    N,T = ts.shape; #ts lenght and n channels
    no_win = int((T - width)/shift) #number of windows
    reds = np.zeros((N, N, no_win)); #init correlation matrix
    for t in range(no_win):
        redundancy_mat = np.zeros((N,N))
        for i1,i2 in combinations(range(N),2):
            red = red_TDMI(ts[i1,t*shift: t*shift+width],ts[i2,t*shift: t*shift+width])
            redundancy_mat[i1,i2] = red
            reds[:,:,t] = redundancy_mat + redundancy_mat.T

    return reds



