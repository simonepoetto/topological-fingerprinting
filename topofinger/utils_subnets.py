import numpy as np
from sklearn.preprocessing import LabelEncoder
import itertools
from .utils import threshold_matrix




def extract_subregion(fc, subnet_idxs, n_sn):
    new_matrix = fc.copy()
    to_mask = np.where(subnet_idxs != n_sn)[0]
    new_matrix = np.delete(new_matrix, to_mask, axis=0)
    new_matrix = np.delete(new_matrix, to_mask, axis=1)
    return new_matrix

def subnet_separation(fc_list, subnet_idxs, subnet_dict):
    subnet_separated = {}
    for key in subnet_dict.keys():
        subnet_separated[key] = []
        n_sn = subnet_dict[key]
        for matrix in fc_list:
            new_matrix = extract_subregion(matrix,subnet_idxs, n_sn)
            subnet_separated[key].append(new_matrix)
    return subnet_separated


def compute_subnet_density(conn_matrix, subnet_list, perc_to_keep=0.5):
    th_conn_matrix = threshold_matrix(conn_matrix, perc_to_keep)
    non_zero = np.count_nonzero(th_conn_matrix)
    label_encoder = LabelEncoder()
    subnet_num_list = label_encoder.fit_transform(subnet_list)
    n = len(set(subnet_num_list))
    subnet_density = np.zeros((n, n))
    combo_subnet = list(itertools.product(range(n), range(n)))
    for i,j in combo_subnet:
        density = np.count_nonzero(th_conn_matrix[subnet_num_list==i][:,subnet_num_list==j])/non_zero
        subnet_density[i,j] = density
    return subnet_density

def compute_subnet_mean(conn_matrix, subnet_list):
    label_encoder = LabelEncoder()
    subnet_num_list = label_encoder.fit_transform(subnet_list)
    n = len(set(subnet_num_list))
    subnet_mean = np.zeros((n, n))
    combo_subnet = list(itertools.product(range(n), range(n)))
    for i,j in combo_subnet:
        subnet_mean[int(i),int(j)] = conn_matrix[subnet_num_list == i][:, subnet_num_list == j].mean()
    return subnet_mean
    