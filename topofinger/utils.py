import numpy as np
import scipy.sparse
from itertools import combinations
import shutil

import pandas as pd

import sys
import os

from itertools import combinations

from tqdm.notebook import tqdm

import random

from sklearn.preprocessing import LabelEncoder

import copy

        
def mean_columns(matrix, n):
    new_columns = [np.mean(matrix[:, i:i+n], axis=1) for i in range(0, matrix.shape[1], n)]
    new_matrix = np.column_stack(new_columns)
    return new_matrix


def sequential_corr_matrices(tt,width,shift):
    T, N = tt.shape; #ts lenght and n channels
    no_win = int((T - width)/shift)+1 #number of windows
    corrs = np.zeros((N, N, no_win)); #init correlation matrix
    for t in range(no_win):
        corrs[:,:,t] = np.corrcoef(tt[t*shift: t*shift+width].T);
    return corrs



def threshold_matrix(matrix, perc_to_keep, zero_diag=True, nan=False):
    """
    Threshold a square matrix to keep only the highest percentage of values.
    
    Args:
        matrix (numpy.ndarray): The input square matrix.
        percentage (float): The percentage of the highest values to keep (0 to 100).
        
    Returns:
        numpy.ndarray: The thresholded matrix with non-top values set to zero.
    """
    if not isinstance(matrix, np.ndarray) or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square numpy.ndarray")

    if perc_to_keep < 0 or perc_to_keep > 100:
        raise ValueError("Percentage should be in the range [0, 100]")
    
    matrix = np.abs(matrix)
    
    if zero_diag:
        #Put 0 on the diagonal
        np.fill_diagonal(matrix,0)
        
    th_value = np.percentile(matrix.flatten(), 100 - perc_to_keep)
    thresholded_matrix = np.where(matrix >= th_value, matrix, 0)    

    if nan:
        thresholded_matrix = np.where(thresholded_matrix == 0, np.nan, thresholded_matrix)
    
    return thresholded_matrix





def apply_threshold_to_nested_dict(d, perc_to_keep):
    """
    Apply the threshold_matrix function to every square matrix within a nested dictionary.

    Args:
        d (dict): The nested dictionary containing square matrices.
        percentage (float): The percentage of the highest values to keep (0 to 100).
        
    Returns:
        dict: The modified nested dictionary with thresholded matrices.
    """
    new_d = copy.deepcopy(d)
    for key, value in new_d.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            new_d[key] = apply_threshold_to_nested_dict(value, perc_to_keep)
        elif isinstance(value, np.ndarray) and value.shape[0] == value.shape[1]:
            # Apply threshold_matrix function to square matrices
            new_d[key] = threshold_matrix(value, perc_to_keep)
    return new_d

def distance_matrix_from_list_of_matrices(matrix_list):
    """
    Computes the matrix of pairwise distances of a list of matrices.
    The distance is computed as the Fobenious distance between the flattened upper triangle part of the matrices

    Parameters:
    -----------
    matrix_list : list
        A list of matrices.

    Returns:
    --------
    dist_mat : numpy.ndarray
        A distance matrix of shape (n, n), where n is the length of matrix_list.
    """
    n = len(matrix_list)
    dist_mat = np.zeros((n,n))
    for i,j in combinations(range(n),2):
        F = matrix_list[i]
        G = matrix_list[j]
        #if scipy.sparse.issparse(F):
        #   F = np.array(F.todense())
        #  G = np.array(G.todense())
        d = np.linalg.norm(F-G) / np.linalg.norm(F+G)
        dist_mat[i,j] = d
    
    dist_mat = dist_mat+dist_mat.T
    
    return dist_mat





def random_adj_matrix(matrix, percent_zero=0.96):
    matrix = (matrix+matrix.T)/2
    indices = np.random.choice(matrix.shape[1]*matrix.shape[0], replace=False, size=int(matrix.shape[1]*matrix.shape[0]*percent_zero))
    matrix[np.unravel_index(indices, matrix.shape)] = 0 
    return matrix



def dict_of_dict_to_dataframe(dictionary):
    """
    Given a dictionary of dictionaries, returns a pandas dataframe with the dictionary keys as column values.
    
    Args:
    - dictionary: A dictionary of dictionaries
    
    Returns:
    - A pandas dataframe with the dictionary keys as column values
    """
    # Initialize an empty list to store the flattened dictionaries
    flattened_dict = []
    
    # Loop through the keys of the outer dictionary
    for key1 in dictionary.keys():
        # Loop through the keys of the inner dictionary
        for key2 in dictionary[key1].keys():
            # Append a flattened dictionary to the list of flattened dictionaries
            flattened_dict.append({
                'subject': key1,
                'session': key2,
                'TS': dictionary[key1][key2]  # Reshape the value to a 1D array
            })
            
    # Create a pandas dataframe from the flattened dictionary
    df = pd.DataFrame(flattened_dict)
    
    # Reshape the "Value" column to a 2D array
    #df['ts'] = df['Value'].apply(lambda x: x.reshape(-1, 1))
    
    # Split the "Value" column into separate columns
    #df = pd.concat([df[['Column 1', 'Column 2']], pd.DataFrame(df['Value'].to_list(), index=df.index)], axis=1)
    #df = df.drop(columns=['Value'])
    
    return df



def dict_of_dict_to_dataframe_windows(dictionary):
    """
    Given a dictionary of dictionaries, returns a pandas dataframe with the dictionary keys as column values.
    
    Args:
    - dictionary: A dictionary of dictionaries
    
    Returns:
    - A pandas dataframe with the dictionary keys as column values
    """
    # Initialize an empty list to store the flattened dictionaries
    flattened_dict = []
    
    # Loop through the keys of the outer dictionary
    for key1 in dictionary.keys():
        # Loop through the keys of the inner dictionary
        for key2 in dictionary[key1].keys():
            for key3 in dictionary[key1][key2].keys():
                # Append a flattened dictionary to the list of flattened dictionaries
                flattened_dict.append({
                    'windows': key1,
                    'subject': key2,
                    'session':key3,
                    'scaffolds': dictionary[key1][key2][key3]  # Reshape the value to a 1D array
                })
                
    # Create a pandas dataframe from the flattened dictionary
    df = pd.DataFrame(flattened_dict)
    
    # Reshape the "Value" column to a 2D array
    #df['ts'] = df['Value'].apply(lambda x: x.reshape(-1, 1))
    
    # Split the "Value" column into separate columns
    #df = pd.concat([df[['Column 1', 'Column 2']], pd.DataFrame(df['Value'].to_list(), index=df.index)], axis=1)
    #df = df.drop(columns=['Value'])
    
    return df


def get_nonzero_indices(matrix):
    """
    Given a matrix, returns a list of tuples of indices where the matrix is nonzero.
    
    Args:
    - matrix: A 2D list or numpy array
    
    Returns:
    - A list of tuples representing the indices of nonzero elements in the matrix
    """
    # Initialize an empty list to store the indices of nonzero elements
    nonzero_indices = []
    
    # Loop through the rows and columns of the matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            # If the element at (i, j) is nonzero, append its indices to the list
            if matrix[i][j] != 0:
                nonzero_indices.append((i, j))
                
    return nonzero_indices


def get_values_and_random(matrix, indices):
    values = [matrix[i, j] for i, j in indices]
    random_values = []

    for _ in range(len(indices)):
        random_row = random.randint(0, matrix.shape[0] - 1)
        random_col = random.randint(0, matrix.shape[1] - 1)
        random_value = matrix[random_row, random_col]
        random_values.append(random_value)

    return values, random_values

def get_values_random_and_minmax(matrix, indices):
    values = [matrix[i, j] for i, j in indices]
    random_values = []
    
    N = len(indices)

    for _ in range(N):
        random_row = random.randint(0, matrix.shape[0] - 1)
        random_col = random.randint(0, matrix.shape[1] - 1)
        random_value = matrix[random_row, random_col]
        random_values.append(random_value)
        
        flattened = matrix.flatten()
        filtered = flattened[(flattened != 0.0) & (flattened != 1.0)]
        ascending_sorted = np.sort(filtered)

        min_list = ascending_sorted[:N].tolist()
        max_list = ascending_sorted[-N:].tolist()

    return values, random_values, min_list, max_list



def get_values_and_random_at_scaffold(corr_matrix,scaff_matrix):
    
    indices = get_nonzero_indices(scaff_matrix)
    
    return get_values_and_random(corr_matrix,indices)

def get_values_random_and_minmax_at_scaffold(corr_matrix,scaff_matrix):
    
    indices = get_nonzero_indices(scaff_matrix)
    
    return get_values_random_and_minmax(corr_matrix,indices)




def get_subnets_idxs(file):
    regions_list = []
    with open(file,'r') as f:
        for line in f:
            elements = line.strip().split(' ')
            regions_list.append(elements[4])
        
    regions7 = [reg.split('_')[2] for reg in regions_list]
    
    label_encoder = LabelEncoder()

    regions7_numbers = label_encoder.fit_transform(regions7)
    dict_regions7 = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    return regions7_numbers, dict_regions7

#### separate RL and LR

def split_folder(input_folder, output_folder1, output_folder2,endswith='hp.csv'):
    # Create output folders if they don't exist
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)
    
    # Get the list of files in the input folder
    files = os.listdir(input_folder)
    
    for file in files:
        file_path = os.path.join(input_folder, file)
        if os.path.isfile(file_path) and file.endswith(endswith):
            # Move the file to the first output folder
            shutil.move(file_path, os.path.join(output_folder1, file))
        elif os.path.isfile(file_path):
            # Move the file to the second output folder
            shutil.move(file_path, os.path.join(output_folder2, file))
        else:
            pass
    