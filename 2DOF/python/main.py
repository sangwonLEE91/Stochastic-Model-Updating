import timeit
import numpy as np
import torch
from VAE_residual_simple import VAE
from REMC_func import initial_sample, replica_exchange

Temp = np.array([1, 2, 4, 8])
n_replica = len(Temp)
iteration = 100
n_exchange = 1000
n_par = 3
channel = 2
trans_sigma = np.array([0.1, 0.1, 0.05])
range_theta = [[0, 2], [0, 2], [0, 0.6]]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device = ', device)
# data
epoch = 1000
z_dim = 3

for test in range(20):
    for n_obs in [4, 8, 16, 32, 64, 128]:
        for n_sim in [16, 32, 64, 128]:
            print(f'--------  n_sim = {n_sim}, n_obs = {n_obs}, test = {test} ---------------')
            obs = np.load(f'../observation/obs_t{test+1}.npy')[:n_obs, :, :512]
            torch_root = f'../torch_model/z{z_dim}_e{epoch}.pth'
            save_root = f'../result/samples_REMC_z{z_dim}_e{epoch}_o{n_obs}_s{n_sim}_t{test+1}.npy'
            net = VAE(z_dim, channel, 512).to(device)
            theta_init, _, z_D, z, net = initial_sample(net, n_par, device, torch_root, obs, n_obs, '../dataset/', n_replica)
            start_time = timeit.default_timer()
            MCMC_par = [range_theta, device, net, trans_sigma, theta_init, z_D, z, iteration, Temp, n_obs, n_sim]
            replica_exchange(save_root, n_exchange, MCMC_par, n_par, False)
            end_time = timeit.default_timer()
            print('total computing time :', end_time - start_time, '(sec)')
