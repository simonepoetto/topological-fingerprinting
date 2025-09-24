import numpy as np
from pingouin import compute_effsize
from scipy.stats import zscore
from itertools import combinations

def compute_idiff(sim_matrix, kind='corr'):
    i_self = np.mean(np.diag(sim_matrix))
    i_others = np.mean(sim_matrix[np.where(~np.eye(sim_matrix.shape[0], dtype=bool))])
    if kind == 'dist':
        idiff = i_others - i_self
    elif kind == 'corr':
        idiff = i_self - i_others
    return idiff, i_self, i_others

def compute_effect_size(sim_matrix, kind='corr'):
    inside_corr = np.diag(sim_matrix)
    outside_corr = sim_matrix[np.where(~np.eye(sim_matrix.shape[0], dtype=bool))]
    if kind == 'dist':
        effect_size = compute_effsize(outside_corr, inside_corr)
    elif kind == 'corr':
        effect_size = compute_effsize(inside_corr, outside_corr)
    return effect_size

def compute_success_rate(sim_matrix, kind='corr'):
    n_sub = sim_matrix.shape[0]
    correct2 = 0
    #guess s2
    for i,sub in enumerate(sim_matrix):
        if kind == 'dist':
            win = np.argmin(sub)
        elif kind =='corr':
            win = np.argmax(sub)
        if win==i:
            correct2 += 1
    correct1 = 0
    #guess s1
    for i,sub in enumerate(sim_matrix.T):
        if kind == 'dist':
            win = np.argmin(sub)
        elif kind =='corr':
            win = np.argmax(sub)
        if win==i:
            correct1 += 1
    return np.round(correct1/n_sub,2)*100, np.round(correct2/n_sub,2)*100

def compute_effect_size_individuals(dist_matrix, n_subject):
    n_tot = dist_matrix.shape[0]
    n_win = n_tot//n_subject
    effect_sizes = []
    for n in range(n_subject):
        inside_corr = dist_matrix[n*n_win:n*n_win+n_win,n*n_win:n*n_win+n_win].flatten()
                
        outside_row = dist_matrix[n*n_win:n*n_win+n_win,:]
        outside_row = np.delete(outside_row, np.s_[n*n_win:n*n_win+n_win], axis=1)
        
        outside_columns = dist_matrix[:,n*n_win:n*n_win+n_win]
        outside_columns = np.delete(outside_columns, np.s_[n*n_win:n*n_win+n_win], axis=0)
        
        outside_corr = np.concatenate((outside_row.flatten(), outside_columns.flatten()))
        
        effect_size = compute_effsize(inside_corr,outside_corr)
        effect_sizes.append(effect_size)
    return effect_sizes

def compute_group_consistency(conn_dict, z=False):
    m = list(conn_dict.values())[0][0].shape[0]
    if z:
        s0 = np.array([zscore(sub[0],axis=None).flatten() for sub in conn_dict.values()])
        s1 = np.array([zscore(sub[1],axis=None).flatten() for sub in conn_dict.values()])
    else:
            s0 = np.array([sub[0].flatten() for sub in conn_dict.values()])
            s1 = np.array([sub[1].flatten() for sub in conn_dict.values()])
    group_consistency = (s0*s1).mean(axis=0).reshape(m,m)
    return group_consistency

def compute_differential_power(conn_dict, z=False):
    n = len(conn_dict.keys())
    if z:    
        s0 = np.array([zscore(sub[0],axis=None) for sub in conn_dict.values()])
        s1 = np.array([zscore(sub[1],axis=None) for sub in conn_dict.values()])
    else:
        s0 = np.array([sub[0] for sub in conn_dict.values()])
        s1 = np.array([sub[1] for sub in conn_dict.values()])
    m = s0[0].shape[0]
    subject_DP = []
    for i in range(s0.shape[0]):
        edges_ii = (s0[i]*s1[i])
        s0t = np.delete(s0,i,0)
        s1t = np.delete(s1,i,0)
        edges_ij = (s0[i]*s1t)
        edges_ji = (s1[i]*s0t)
        subject_i = np.zeros((m,m))
        for c,r in combinations(range(m),2):
            if (edges_ii[c,r] == 0) & (edges_ij[:,c,r].sum() == 0) & (edges_ji[:,c,r].sum() == 0):
                subject_i[c,r] = 1
            else:
                subject_i[c,r] = (np.sum(edges_ij[:,c,r]>edges_ii[c,r]) + np.sum(edges_ji[:,c,r]>edges_ii[c,r]))/((n-1)*2)        
        subject_DP.append((subject_i + subject_i.T)/2)
    differential_power = -np.log(np.array(subject_DP)).sum(axis=0)
    differential_power[np.isinf(differential_power)] = np.log(0.1/((n-1)*2))
    
    return differential_power


    