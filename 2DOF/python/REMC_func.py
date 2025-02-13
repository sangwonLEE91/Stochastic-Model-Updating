import copy
import timeit
import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset
from VAE_residual_simple import VAE
from simulator import simulator_only, noise_nonzero
from data_loader import dataset
from lee import calculate_likelihood
import multiprocessing as mp
import os
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, data):
        self.x_data = data.tolist()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        return x


def transition(theta, sigma, j):
    theta_new = copy.deepcopy(theta)
    for i in range(theta.shape[1]):
        theta_new[j, i] = np.random.normal(theta[j, i], sigma[j], (1,))[0]
    return theta_new


def cal_loglikelihood(z_D, z_theta, z):
    z_dim = len(z_D[0])
    LH = 0.
    for i in range(z_dim):
        mu1 = z_D[0][i]
        mu2 = z_theta[0][i]
        mu3 = z[0][i]
        vr1 = z_D[1][i]
        vr2 = z_theta[1][i]
        vr3 = z[1][i]
        c0 = np.sqrt(vr3 / (2 * np.pi * vr1 * vr2))
        a = (1 / (2 * vr1)) + (1 / (2 * vr2)) - (1 / (2 * vr3))
        b = -(mu1 / vr1) - (mu2 / vr2) + (mu3 / vr3)
        c = (mu1 ** 2 / (2 * vr1)) + (mu2 ** 2 / (2 * vr2)) - (mu3 ** 2 / (2 * vr3))
        LH += np.log(c0) + ((b ** 2 - 4 * a * c) / (4 * a)) + np.log(np.sqrt(np.pi / a))
    return LH


def prior(theta, prior_pdf):
    p = prior_pdf[0].pdf(theta[:2])
    p *= prior_pdf[1].pdf(theta[2])
    return p


def likelihood(device, theta, net, z_D, z, T, n_obs, n_sim):
    # #print(theta, z_theta_mu,z_theta_var)
    LH = np.zeros((n_obs,))
    TF = []
    for j in range(n_sim):
        theta_noise = noise_nonzero(theta[:2], theta[2])
        tf = simulator_only(theta_noise)
        TF.append(tf)
    TF = torch.tensor(TF).to(device)
    _, z_theta_mu, z_theta_var = net.enc(TF.view(-1, TF.shape[1], 1, TF.shape[2]))

    z_theta_mu_np = z_theta_mu.cpu().detach().numpy()
    z_theta_var_np = z_theta_var.cpu().detach().numpy()

    z_dim = len(z_D[0][0])
    z_D_mu = np.array([z_D[i][0] for i in range(n_obs)])
    z_D_var = np.array([z_D[i][1] for i in range(n_obs)])

    LH = calculate_likelihood(
        n_obs, n_sim, z_dim, z_D_mu, z_D_var, z_theta_mu_np, z_theta_var_np, np.array(z[0]), np.array(z[1]))

    if np.any(LH[:] == 0):
        LH_prod = -10 ** 8
    else:
        LH_prod = np.sum(np.log(LH))

    return LH_prod / T


def likelihood_R(device, theta, net, z_D, z, T, i_nonz, n_obs, n_sim):
    # net에 넣기
    LH = np.zeros((len(i_nonz), n_obs))
    LH_out = np.zeros((len(i_nonz),))
    z_dim = len(z_D[0][0])
    z_D_mu = np.array([z_D[i][0] for i in range(n_obs)])
    z_D_var = np.array([z_D[i][1] for i in range(n_obs)])
    for k in range(len(i_nonz)):
        TF = []
        for j in range(n_sim):
            theta_noise = noise_nonzero(theta[k][:2], theta[k][2])
            TF.append(simulator_only(theta_noise))
        TF = torch.tensor(TF).to(device)

        _, z_theta_mu, z_theta_var = net.enc(TF.view(-1, TF.shape[1], 1, TF.shape[2]))
        z_theta_mu_np = z_theta_mu.cpu().detach().numpy()
        z_theta_var_np = z_theta_var.cpu().detach().numpy()

        LH[k, :] = calculate_likelihood(
            n_obs, n_sim, z_dim, z_D_mu, z_D_var, z_theta_mu_np, z_theta_var_np, np.array(z[0]), np.array(z[1]))

        if np.any(LH[k, :] == 0):
            LH_out[k] = -10e8
        else:
            LH_out[k] = np.sum(np.log(LH[k, :]))
    return LH_out / T[i_nonz]


# Input likelihood
def acceptance(x, x_new):
    if (x_new >= x):
        return True
    else:
        accept = np.random.uniform(0, 1)
        return (np.log(accept) < x_new - x)


def gibbs_mcmc(range_theta, device, net, sigma, theta_init, z_D_pdf, z_pdf, iterations, T, n_obs, n_sim):
    theta_old = theta_init
    n_replica = len(T)
    n_c = len(theta_old)
    lik_theta_old = np.zeros((n_replica,))
    lik_theta_new = np.zeros((n_replica,))

    for i_re in range(n_replica):
        lik_theta_old[i_re] = likelihood(device, theta_old[:, i_re], net, z_D_pdf, z_pdf, T[i_re], n_obs, n_sim)

    a = lik_theta_old  # * prior(theta_old, prior_pdf)
    accepted = [theta_old]
    i_iteration = 0
    while i_iteration < iterations:
        for j in range(n_c):
            theta_new = transition(theta_old, sigma, j)
            theta_nonz = []
            i_zero = []
            i_nonz = []
            for i in range(n_replica):
                if (theta_new[j, i] <= range_theta[j][0] or theta_new[j, i] >= range_theta[j][1]):
                    lik_theta_new[i] = -(10 ** 15)
                    i_zero.append(i)
                else:
                    i_nonz.append(i)
                    theta_nonz.append(theta_new[:, i])

            if i_nonz:
                lik_theta_new[i_nonz] = likelihood_R(device, theta_nonz, net, z_D_pdf, z_pdf, T, i_nonz, n_obs, n_sim)

            b = lik_theta_new  # * prior(theta_new, prior_pdf)
            for i in range(n_replica):
                if (acceptance(a[i], b[i])):
                    theta_old[:, i] = theta_new[:, i]
                    a[i] = b[i]

            accepted.append(theta_old)
            i_iteration += 1
    return np.array(accepted)[:iterations, :, :]


def replica_exchange(save_root, n_exchange, par, npar, pretrained):
    range_theta, device, net, transition_sigma, theta_initial, z_D, z, iteration, Temperature, n_obs, n_sim = par
    accept_ar = np.zeros((iteration * n_exchange, npar))
    n_replica = len(Temperature)
    start = 0
    if pretrained:
        pre_sample = np.load(save_root)
        start = int(pre_sample.shape[0] / iteration)
        for j in range(n_replica):
            theta_initial[:, j] = pre_sample[-1, :]
        accept_ar[:pre_sample.shape[0], :] = pre_sample

    exchange = []

    for jj in range(start, n_exchange):
        accepted = gibbs_mcmc(range_theta, device, net, transition_sigma, theta_initial, z_D, z,
                              iteration, Temperature, n_obs, n_sim)
        accept_ar[jj * iteration:(jj + 1) * iteration, :] = accepted[:, :, 0]  # save only first replica
        theta_initial = copy.deepcopy(accepted[-1, :, :])
        if (jj + 1) % 100 == 0:
            np.save(save_root, accept_ar[:(jj + 1) * iteration, :])  # t replica save only

        theta = theta_initial
        if jj % 2 == 0:
            if n_replica % 2 == 0:
                i_k = np.arange(0, n_replica, 2)
            else:
                i_k = np.arange(0, n_replica - 1, 2)
        else:
            i_k = np.arange(1, n_replica - 1, 2)
        for k in i_k:
            a = copy.deepcopy(theta[:, k])
            b = copy.deepcopy(theta[:, k + 1])

            lik_o1 = likelihood(device, a, net, z_D, z, Temperature[k], n_obs, n_sim)
            lik_o2 = likelihood(device, b, net, z_D, z, Temperature[k + 1], n_obs, n_sim)
            lik_old = lik_o1 + lik_o2
            lik_n1 = lik_o2 * Temperature[k + 1] / Temperature[k]
            lik_n2 = lik_o1 * Temperature[k] / Temperature[k + 1]
            lik_new = lik_n1 + lik_n2
            if (acceptance(lik_old, lik_new)):
                theta[:, k] = b
                theta[:, k + 1] = a
                exchange.append(str(k) + str(k + 1))

        theta_initial = theta


def initial_sample(net, npar, device, net_root, response, n_obs, folder, n_replica):
    net.load_state_dict(torch.load(net_root))
    net.eval()
    n_initial = 30
    z_p = []
    valid_l = folder + 'valid_label.npy'
    valid_d = folder + 'valid_data.npy'
    valid_loader, _, valid_label = dataset(valid_d, valid_l, batch_size=64, shuffle=False)

    for counter, wave in enumerate(valid_loader):
        wave = wave.to(device)
        zz_p, mu, var = net.enc(wave)
        z_p += zz_p.cpu().detach().numpy().tolist()

    z_p = np.array(z_p)
    z_mu = np.mean(z_p, axis=0)
    z_s2 = np.var(z_p, axis=0)
    z = [z_mu, z_s2]

    z_D = []
    tf = torch.tensor(response.tolist()).to(device)
    _, z_D_mu, z_D_var = net.enc(tf.view(-1, tf.shape[1], 1, tf.shape[2]))
    for j in range(n_obs):
        z_D.append([z_D_mu[j].cpu().detach().numpy(), z_D_var[j].cpu().detach().numpy()])

    distance = np.mean((z_p - np.mean(np.array(z_D)[:, 0, :], axis=0)) ** 2, axis=1)
    initial_list = valid_label[np.argsort(distance)[:n_initial], :]
    theta_init = np.zeros((npar, n_replica))
    for j in range(n_replica):
        theta_init[:, j] = initial_list[np.random.randint(0, n_initial), :npar]
    return theta_init, initial_list, z_D, z, net
