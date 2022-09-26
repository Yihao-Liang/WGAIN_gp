'''Utility functions for GAIN and WGAIN_gp.

- rounding: Handlecategorical variables after imputation
- rmse_loss: Evaluate imputed data in terms of RMSE
- xavier_init: Xavier initialization
- sample_M: binary_sampler
- sample_Z: sample uniform random variables
- sample_batch_index: sample random batch index
'''

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def rounding(imputed_data, data_x):
    '''Round imputed data for categorical variables.

    Parameters:
      - imputed_data: imputed data
      - data_x: original data with missing values

    Returns:
      - rounded_data: rounded imputed data
    '''

    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])

    return rounded_data


def rmse_loss(ori_data, imputed_data, data_m):
    '''Compute RMSE loss between ori_data and imputed_data

    Parameters:
      - ori_data: original data without missing values
      - imputed_data: imputed data
      - data_m: indicator matrix for missingness

    Returns:
      - rmse: Root Mean Squared Error
    '''

    scaler = MinMaxScaler()
    ori_data = scaler.fit_transform(ori_data)
    imputed_data = scaler.fit_transform(X=imputed_data)

    # Only for missing values
    nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)

    rmse = np.sqrt(nominator / float(denominator))

    return rmse


def xavier_init(size):
    '''Xavier initialization.

    Parameters:
      - size: vector size

    Returns:
      - initialized random vector.
    '''
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size=size, scale=xavier_stddev)


def sample_M(m, n, p):
    '''Generate Mask Vector and Hint Vector
    
    Parameters:
        - m: the number of rows
        - n: the number of columns
        - p: probability of 1 (missing rate)
        
    Returns:
        - binary_random_matrix: generated binary random matrix.
    '''

    A = np.random.uniform(0., 1., size=[m, n])
    B = A > p
    C = 1. * B
    return C


def sample_Z(m, n):
    '''Sample random variables.
    
    Parameters:
        - m: the number of rows
        - n: the number of columns
        
    Returns:
        - random_matrix: generated random matrix.
    '''
    return np.random.uniform(0, 0.01, size=[m, n])  


def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.

    Parameters:
      - total: total number of samples
      - batch_size: batch size

    Returns:
      - batch_idx: batch index
    '''
    total_idx = np.random.permutation(total)    # 对序列total随机排序
    batch_idx = total_idx[:batch_size]          # 获取前batch_size个值的数组
    return batch_idx