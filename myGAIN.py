# myGAIN
# This is a reprocution of GAIN (Generative Adversarial Imputation Nets) based on PyTorch.
# Written by Yihao Liang
# Date: Sep 19th 2022
# Reference: 
# - J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets," ICML, 2018.
# - [GAIN by jsyoon0823](https://github.com/jsyoon0823/GAIN)

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import *


def gain(data_x):
    '''Impute missing values in data_x

    Parameters:
      - data_x: original data with missing values
    Returns:
      - imputed_data: imputed data
    '''
    # Define mask matrix
    data_m = 1. - np.isnan(data_x)
    h_dim = int(dim)                # Hidden state dimensions
    
    # Normalization
    norm_data = scaler.fit_transform(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)  # NaN -> 0
    
    # GAIN architecture
    # Discriminator variables
    if use_gpu is True:
        D_W1 = torch.tensor(xavier_init([dim * 2, h_dim]), requires_grad=True, device="cuda")  # Data + Hint as inputs
        D_b1 = torch.tensor(np.zeros(shape=[h_dim]), requires_grad=True, device="cuda")

        D_W2 = torch.tensor(xavier_init([h_dim, h_dim]), requires_grad=True, device="cuda")
        D_b2 = torch.tensor(np.zeros(shape=[h_dim]), requires_grad=True, device="cuda")

        D_W3 = torch.tensor(xavier_init([h_dim, dim]), requires_grad=True, device="cuda")
        # Output is multi-variate
        D_b3 = torch.tensor(np.zeros(shape=[dim]), requires_grad=True, device="cuda")
    else:
        # Data + Hint as inputs
        D_W1 = torch.tensor(xavier_init([dim * 2, h_dim]), requires_grad=True)
        D_b1 = torch.tensor(np.zeros(shape=[h_dim]), requires_grad=True)

        D_W2 = torch.tensor(xavier_init([h_dim, h_dim]), requires_grad=True)
        D_b2 = torch.tensor(np.zeros(shape=[h_dim]), requires_grad=True)

        D_W3 = torch.tensor(xavier_init([h_dim, dim]), requires_grad=True)
        # Output is multi-variate
        D_b3 = torch.tensor(np.zeros(shape=[dim]), requires_grad=True)

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    if use_gpu is True:
        G_W1 = torch.tensor(xavier_init([dim * 2, h_dim]), requires_grad=True, device="cuda")  # Data + Mask as inputs (Random Noises are in Missing Components)
        G_b1 = torch.tensor(np.zeros(shape=[h_dim]), requires_grad=True, device="cuda")

        G_W2 = torch.tensor(xavier_init([h_dim, h_dim]), requires_grad=True, device="cuda")
        G_b2 = torch.tensor(np.zeros(shape=[h_dim]), requires_grad=True, device="cuda")

        G_W3 = torch.tensor(xavier_init([h_dim, dim]), requires_grad=True, device="cuda")
        G_b3 = torch.tensor(np.zeros(shape=[dim]), requires_grad=True, device="cuda")
    else:
        G_W1 = torch.tensor(xavier_init([dim * 2, h_dim]), requires_grad=True)  # Data + Mask as inputs (Random Noises are in Missing Components)
        G_b1 = torch.tensor(np.zeros(shape=[h_dim]), requires_grad=True)

        G_W2 = torch.tensor(xavier_init([h_dim, h_dim]), requires_grad=True)
        G_b2 = torch.tensor(np.zeros(shape=[h_dim]), requires_grad=True)

        G_W3 = torch.tensor(xavier_init([h_dim, dim]), requires_grad=True)
        G_b3 = torch.tensor(np.zeros(shape=[dim]), requires_grad=True)

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    # GAIN functions
    # Generator
    def generator(x, m):
        # Concatenate Data and Mask
        inputs = torch.cat(dim=1, tensors=[x, m])
        G_h1 = F.relu(torch.matmul(inputs, G_W1) + G_b1)
        G_h2 = F.relu(torch.matmul(G_h1, G_W2) + G_b2)
        # MinMax normalized output
        G_prob = torch.sigmoid(torch.matmul(G_h2, G_W3) + G_b3)
        return G_prob

    # Discriminator
    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = torch.cat(dim=1, tensors=[x, h])
        D_h1 = F.relu(torch.matmul(inputs, D_W1) + D_b1)
        D_h2 = F.relu(torch.matmul(D_h1, D_W2) + D_b2)
        D_logit = torch.matmul(D_h2, D_W3) + D_b3
        D_prob = torch.sigmoid(D_logit)
        return D_prob

    # GAIN Loss
    def discriminator_loss(M, X, H):
        # Generator
        G_sample = generator(X, M)
        # Combine with original data
        X_hat = X * M + G_sample * (1 - M)
        # Discriminator
        D_prob = discriminator(X_hat, H)
        # Loss
        D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) +(1 - M) * torch.log(1. - D_prob + 1e-8))
        return D_loss

    def generator_loss(X, M, H):
        # Structure
        # Generator
        G_sample = generator(X, M)
        # Combine with original data
        X_hat = X * M + G_sample * (1 - M)
        # Discriminator
        D_prob = discriminator(X_hat, H)
        # Loss
        G_loss1 = -torch.mean((1 - M) * torch.log(D_prob + 1e-8))
        MSE_train_loss = torch.mean((M * X - M * G_sample) ** 2) / torch.mean(M)

        G_loss = G_loss1 + alpha * MSE_train_loss
        return G_loss, MSE_train_loss

    # optimizer Adam
    optimizer_D = torch.optim.Adam(params=theta_D)
    optimizer_G = torch.optim.Adam(params=theta_G)

    # Start Iterations
    for it in tqdm(range(iterations + 1)):
        # Sample batch
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]        # X size: (128,57)
        M_mb = data_m[batch_idx, :]             # M size: (128,57)
        Z_mb = sample_Z(batch_size, dim)        # Z
        
        # Hint Vector
        H_mb_temp = sample_M(batch_size, dim, 1 - hint_rate)
        H_mb = M_mb * H_mb_temp                 

        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # X~

        # convert to tensor
        if use_gpu is True:
            X_mb = torch.tensor(X_mb, device="cuda")
            M_mb = torch.tensor(M_mb, device="cuda")
            H_mb = torch.tensor(H_mb, device="cuda")
        else:
            X_mb = torch.tensor(X_mb)
            M_mb = torch.tensor(M_mb)
            H_mb = torch.tensor(H_mb)

        optimizer_D.zero_grad()
        D_loss_curr = discriminator_loss(M=M_mb, X=X_mb, H=H_mb)
        D_loss_curr.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        G_loss_curr, MSE_train_loss_curr = generator_loss(X=X_mb, M=M_mb, H=H_mb)
        G_loss_curr.backward()
        optimizer_G.step()

        # Intermediate Losses
        if it % (iterations/10) == 0:
            print('Iter: {}'.format(it), end='\t')
            print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())))

    # Return imputed data
    Z_mb = sample_Z(no, dim)
    M_mb = data_m
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
    
    # convert to tensor
    if use_gpu is True:
        X_mb = torch.tensor(X_mb, device='cuda')
        M_mb = torch.tensor(M_mb, device='cuda')
    else:
        X_mb = torch.tensor(X_mb)
        M_mb = torch.tensor(M_mb)

    imputed_data = generator(X_mb, M_mb)

    if use_gpu is True:
        imputed_data = imputed_data.cpu().detach().numpy()
    else:
        imputed_data = imputed_data.detach().numpy()

    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

    # Renormalization
    imputed_data = scaler.inverse_transform(imputed_data)

    # Rounding
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data


if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    data_name = 'spam'
    miss_rate = 0.2
    batch_size = 128
    hint_rate = 0.9
    alpha = 100
    iterations = int(np.ceil(10000))
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Importing Dataset
    # %% spam
    file_name = 'spam.csv'
    data_x = pd.read_csv(file_name)
    data_y = data_x['spam'].values
    data_x = data_x.drop('spam', axis=1).values

    # %% breast
    # file_name = 'breast.csv'
    # data_x = pd.read_csv(file_name)
    # class_mapping = {'M': 0, 'B': 1}
    # data_x['Diagnosis'] = data_x['Diagnosis'].map(class_mapping)
    # data_y = data_x['Diagnosis'].values
    # data_x = data_x.drop('Diagnosis', axis=1).values

    # %% credit
    # file_name = 'credit.csv'
    # data_x = pd.read_csv(file_name)
    # data_y = data_x['def. pay. n. m.'].values
    # data_x = data_x.drop('def. pay. n. m.', axis=1).values
    # data_x = data_x.astype(np.float64)
    # data_y = data_y.astype(np.float64)

    # %% letter
    # file_name = 'letter.csv'
    # data_x = pd.read_csv(file_name)
    # data_y = data_x['letter'].values
    # data_x = data_x.drop('letter', axis=1).values
    # data_y = [ord(y) - 65 for y in data_y]
    # data_x = data_x.astype(np.float64)
    # data_y = np.array(data_y).astype(np.float64)


    # %% news
    # file_name = 'news.csv'
    # data_x = pd.read_csv(file_name)
    # data_y = data_x['data_channel_is_world'].values
    # data_x = data_x.drop('data_channel_is_world', axis=1).values

    no, dim = data_x.shape      # (4601,57)
    m_dim = int(dim) 

    # Importing Missing Data
    data_m = sample_M(no, dim, miss_rate)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan
    # return data_x, miss_data_x, data_m

    # Impute missing data
    imputed_data_x = gain(miss_data_x)

    # calculate and print RMSE
    ori_data_x = data_x
    rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
    print("Imputed test data:")
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)}) 
    print(imputed_data_x)
    print('RMSE: ' + str(np.round(rmse, 4)))
    
    # calculate and print AUROC
    imputed_data_x = scaler.fit_transform(imputed_data_x)
    x_train, x_test, y_train, y_test = train_test_split(imputed_data_x, data_y, test_size=0.3, random_state=7)
    breast_lg = LogisticRegression(max_iter=30000)
    breast_lg.fit(x_train, y_train)
    test_pred = breast_lg.predict_proba(x_test)
    auc_score = roc_auc_score(y_test, test_pred[:, 1])
    print('AUROC:', auc_score)