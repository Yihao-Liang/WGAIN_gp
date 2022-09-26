# myGAIN
# This is an improvement of GAIN based on WGAN_gp(Wasserstein GAN with gradient penalty).
# Written by Yihao Liang
# Date: Sep 22nd 2022
# Reference:
# - M. Arjovsky, S. Chintala, L. Bottou, "Wasserstein Generative Adversarial Networks," PMLR, 2017.
# - [GAIN by jsyoon0823](https://github.com/jsyoon0823/GAIN)

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import torch.autograd as autograd
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import *


# gradient penalty
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    eps = Tensor(sample_Z(batch_size, m_dim))
    # Get random interpolation between real and fake samples
    X_interpolates = (eps * real_samples + ((1 - eps) * fake_samples)).requires_grad_(True)
    d_interpolates = D(X_interpolates)
    fake = Variable(Tensor(real_samples.shape[0], m_dim).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=X_interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_norm = torch.sqrt(epsilon + torch.sum(gradients ** 2, dim=1))
    gradient_penalty = lambda_gp * torch.mean((grad_norm - 1) ** 2)
    # gradients = gradients.view(gradients.size(0), -1)
    # gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def WGAIN_gp(data):
    # WGAIN architecture
    # Discriminator variables
    if use_gpu is True:
        D_W1 = torch.tensor(xavier_init([m_dim, m_dim]), requires_grad=True, device="cuda")  
        D_b1 = torch.tensor(np.zeros(shape=[m_dim]), requires_grad=True, device="cuda")

        D_W2 = torch.tensor(xavier_init([m_dim, m_dim]), requires_grad=True, device="cuda")
        D_b2 = torch.tensor(np.zeros(shape=[m_dim]), requires_grad=True, device="cuda")

        D_W3 = torch.tensor(xavier_init([m_dim, m_dim]), requires_grad=True, device="cuda")
        D_b3 = torch.tensor(np.zeros(shape=[m_dim]), requires_grad=True, device="cuda")
    else:
        
        D_W1 = torch.tensor(xavier_init([m_dim, m_dim]), requires_grad=True)
        D_b1 = torch.tensor(np.zeros(shape=[m_dim]), requires_grad=True)

        D_W2 = torch.tensor(xavier_init([m_dim, m_dim]), requires_grad=True)
        D_b2 = torch.tensor(np.zeros(shape=[m_dim]), requires_grad=True)

        D_W3 = torch.tensor(xavier_init([m_dim, m_dim]), requires_grad=True)
        D_b3 = torch.tensor(np.zeros(shape=[m_dim]), requires_grad=True)
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator
    if use_gpu is True:
        # Data + Mask as inputs (Random Noises are in Missing Components)
        G_W1 = torch.tensor(xavier_init([m_dim * 2, m_dim]), requires_grad=True, device="cuda")
        G_b1 = torch.tensor(np.zeros(shape=[m_dim]), requires_grad=True, device="cuda")

        G_W2 = torch.tensor(xavier_init([m_dim, m_dim]), requires_grad=True, device="cuda")
        G_b2 = torch.tensor(np.zeros(shape=[m_dim]), requires_grad=True, device="cuda")

        G_W3 = torch.tensor(xavier_init([m_dim, m_dim]), requires_grad=True, device="cuda")
        G_b3 = torch.tensor(np.zeros(shape=[m_dim]), requires_grad=True, device="cuda")
    else:
        # Data + Mask as inputs (Random Noises are in Missing Components)
        G_W1 = torch.tensor(xavier_init([m_dim * 2, m_dim]), requires_grad=True)
        G_b1 = torch.tensor(np.zeros(shape=[m_dim]), requires_grad=True)

        G_W2 = torch.tensor(xavier_init([m_dim, m_dim]), requires_grad=True)
        G_b2 = torch.tensor(np.zeros(shape=[m_dim]), requires_grad=True)

        G_W3 = torch.tensor(xavier_init([m_dim, m_dim]), requires_grad=True)
        G_b3 = torch.tensor(np.zeros(shape=[m_dim]), requires_grad=True)

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    # GAIN functions
    # Generator
    def generator(z, m):
        # Concatenate Data and Mask
        inputs = torch.cat(dim=1, tensors=[z, m])
        G_h1 = F.relu(torch.matmul(inputs, G_W1) + G_b1)
        G_h2 = F.relu(torch.matmul(G_h1, G_W2) + G_b2)
        G_prob = torch.tanh(torch.matmul(G_h2, G_W3) + G_b3)
        return G_prob
    # Discriminator

    def discriminator(x):
        inputs = x
        D_h1 = F.relu(torch.matmul(inputs, D_W1) + D_b1)
        D_h2 = F.relu(torch.matmul(D_h1, D_W2) + D_b2)
        D_prob = torch.matmul(D_h2, D_W3) + D_b2    # don't use any activation function
        return D_prob

    # GAIN Loss
    def discriminator_loss(X, M, Z):
        # Generator
        G_sample = generator(Z, M)
        # Discriminator
        D_real = discriminator(X)
        D_fake = discriminator(G_sample)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(
            discriminator, (M*X).data, ((1-M)*G_sample).data)
        D_loss = torch.mean(M * D_real) - torch.mean((1 - M)
                                                     * D_fake) + gradient_penalty
        return D_loss

    def generator_loss(X, M, Z):
        # Generator
        G_sample = generator(Z, M)
        # Discriminator
        # D_real = discriminator(X)
        D_fake = discriminator(G_sample)
        # Loss
        G_loss1 = -torch.mean((1 - M) * D_fake)
        MSE_loss = torch.mean((M * X - M * G_sample) ** 2) / torch.mean(M)

        G_loss = G_loss1 + alpha * MSE_loss
        return G_loss, MSE_loss

    data = data.copy()
    data_miss = scaler.fit_transform(data)
    data_mask = 1. - np.isnan(data) 
    data_miss = np.nan_to_num(data_miss, nan=0.00)

    # optimizer
    # optimizer_D = torch.optim.Adam(params=theta_D, lr=lr, betas=(beta_1, beta_2), eps=epsilon,weight_decay=decay)
    # optimizer_G = torch.optim.Adam(params=theta_G, lr=lr, betas=(beta_1, beta_2), eps=epsilon,weight_decay=decay)
    optimizer_D = torch.optim.RMSprop(params=theta_D, lr=lr, weight_decay=decay, momentum=momentum, eps=epsilon)
    optimizer_G = torch.optim.RMSprop(params=theta_G, lr=lr, weight_decay=decay, momentum=momentum, eps=epsilon)

    for it in tqdm(range(iterations+1)):
        for _ in range(n_critic):       # 每训练n_critic次判别器训练一次生成器
            batch_idx = sample_batch_index(total=no, batch_size=batch_size)
            X_mb = data_miss[batch_idx, :]
            M_mb = data_mask[batch_idx, :]
            Z_mb = M_mb * X_mb + (1 - M_mb) * sample_Z(batch_size, m_dim)

            if use_gpu is True:
                X_mb = torch.tensor(X_mb, device="cuda")
                M_mb = torch.tensor(M_mb, device="cuda")
                Z_mb = torch.tensor(Z_mb, device="cuda")
            else:
                X_mb = torch.tensor(X_mb)
                M_mb = torch.tensor(M_mb)
                Z_mb = torch.tensor(Z_mb)

            optimizer_D.zero_grad()
            D_loss_curr = discriminator_loss(X=M_mb, M=X_mb, Z=Z_mb)
            D_loss_curr.backward()
            optimizer_D.step()

        optimizer_G.zero_grad()
        G_loss_curr, MSE_loss_curr = generator_loss(X=X_mb, M=M_mb, Z=Z_mb)
        G_loss_curr.backward()
        optimizer_G.step()

        if it % (iterations/10) == 0:
            tqdm.write(f"Iteration: {it}; "
                       f"MSE_loss: { MSE_loss_curr:.4}")

    # impute data
    Z_all = data_mask * data_miss + (1 - data_mask) * sample_Z(no, m_dim)
    # convert to tensor
    if use_gpu is True:
        Z_all = torch.tensor(Z_all, device='cuda')
        data_mask = torch.tensor(data_mask, device='cuda')
    else:
        Z_all = torch.tensor(Z_all)
        data_mask = torch.tensor(data_mask)

    imputed_data = generator(z=Z_all, m=data_mask)
    if use_gpu is True:
        imputed_data = imputed_data.cpu().detach().numpy()
        data_mask = data_mask.cpu().detach().numpy()
    else:
        imputed_data = imputed_data.detach().numpy()
        data_mask = data_mask.detach().numpy()

    imputed_data = scaler.inverse_transform(data_mask * data_miss + (1 - data_mask) * imputed_data)
    imputed_data = rounding(imputed_data, data)
    
    return imputed_data


if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    n_critic = 5          # number of additional iterations to train the critic
    batch_size = 128
    miss_rate = 0.2
    lambda_gp = 10  # Loss weight for gradient penalty
    iterations = int(np.ceil(10000/5))
    alpha = 100
    lr = 1e-3
    beta_1 = 0.9
    beta_2 = 0.999
    decay = 0.900       # RMSProp optimizer hyper-parameter
    momentum = 0.000    # RMSProp optimizer hyper-parameter
    epsilon = 1e-8

    scaler = MinMaxScaler(feature_range=(0, 1))
    Tensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor

    # Importing Dataset
    # %% spam
    file_name = 'data/spam.csv'
    data_x = pd.read_csv(file_name)
    data_y = data_x['spam'].values
    data_x = data_x.drop('spam', axis=1).values

    # %% breast
    # file_name = 'data/breast.csv'
    # data_x = pd.read_csv(file_name)
    # class_mapping = {'M': 0, 'B': 1}
    # data_x['Diagnosis'] = data_x['Diagnosis'].map(class_mapping)
    # data_y = data_x['Diagnosis'].values
    # data_x = data_x.drop('Diagnosis', axis=1).values

    # %% credit
    # file_name = 'data/credit.csv'
    # data_x = pd.read_csv(file_name)
    # data_y = data_x['def. pay. n. m.'].values
    # data_x = data_x.drop('def. pay. n. m.', axis=1).values
    # data_x = data_x.astype(np.float64)
    # data_y = data_y.astype(np.float64)

    # %% letter
    # file_name = 'data/letter.csv'
    # data_x = pd.read_csv(file_name)
    # data_y = data_x['letter'].values
    # data_x = data_x.drop('letter', axis=1).values
    # data_y = [ord(y) - 65 for y in data_y]
    # data_x = data_x.astype(np.float64)
    # data_y = np.array(data_y).astype(np.float64)

    # %% news
    # file_name = 'data/news.csv'
    # data_x = pd.read_csv(file_name)
    # data_y = data_x['data_channel_is_world'].values
    # data_x = data_x.drop('data_channel_is_world', axis=1).values

    no, dim = data_x.shape
    m_dim = int(dim)    

    data_m = sample_M(no, dim, miss_rate)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan
    # return data_x, miss_data_x, data_m
    
    # Impute missing data
    imputed_data_x = WGAIN_gp(miss_data_x)
    
    # calculate and print RMSE
    ori_data_x = data_x
    rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
    print("Imputed test data:")
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)}) 
    print(imputed_data_x)
    print('RMSE: ' + str(np.round(rmse, 4)))

    # calculate AUROC
    imputed_data_x = scaler.fit_transform(imputed_data_x)
    x_train, x_test, y_train, y_test = train_test_split(imputed_data_x, data_y, test_size=0.3, random_state=7)
    breast_lg = LogisticRegression(max_iter=30000)
    breast_lg.fit(x_train, y_train)
    test_pred = breast_lg.predict_proba(x_test)
    auc_score = roc_auc_score(y_test, test_pred[:, 1])
    print('AUROC:', auc_score)