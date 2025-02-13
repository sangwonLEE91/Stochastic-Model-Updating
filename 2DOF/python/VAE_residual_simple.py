import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# WEB_HOOK_URLは下準備で発行したURLを設定しください

root = str(np.load('C:/root/root.npy'))


# dimension unchanging
def reparameterization(mean, var, device):
    epsilon = torch.randn(mean.shape).to(device)
    return mean + torch.sqrt(var) * epsilon


def gauss_gauss_kl(mean1, var1, mean2, var2):
    epsilon_val = 1e-8  # Small constant to avoid NaN
    _var2 = var2 + epsilon_val
    _kl = torch.log(_var2) - torch.log(var1) \
          + (var1 + (mean1 - mean2) ** 2) / _var2 - 1
    return 0.5 * torch.mean(torch.sum(_kl))


def gauss_unitgauss_kl(mean, var):
    return -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean ** 2 - var))


def rec_loss_norm4D(x, mean, var):
    return -torch.mean(
        torch.sum(-0.5 * ((x - mean) ** 2 / var + torch.log(var) + torch.log(torch.tensor(2 * torch.pi))),
                  dim=(1, 2, 3)))


def rec_loss_norm2D(x, mean, var):
    return -torch.mean(
        torch.sum(-0.5 * ((x - mean) ** 2 / var + torch.log(var) + torch.log(torch.tensor(2 * torch.pi))),
                  dim=1))

class resblock_enc(nn.Module):
    def __init__(self, in_channels, out_channels, pooling_size):
        super(resblock_enc, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.zeros_(self.conv1.bias)

        self.model = nn.Sequential(
            nn.ReLU(),
            spectral_norm(self.conv1),
            nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_size, padding=0)
        )

        self.bypass_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                     bias=True)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.)
        nn.init.zeros_(self.bypass_conv.bias)

        self.bypass = nn.Sequential(
            spectral_norm(self.bypass_conv),
            nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_size, padding=0)
        )

    def forward(self, x):
        out_x = self.model(x) + self.bypass(x)

        return out_x


class first_resblock_enc(nn.Module):
    def __init__(self, in_channels, out_channels, pooling_size):
        super(first_resblock_enc, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

        self.model = nn.Sequential(
            spectral_norm(self.conv1),
            nn.ReLU(),
            spectral_norm(self.conv2),
            nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_size, padding=0)
        )

        self.bypass_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                     bias=True)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.)
        nn.init.zeros_(self.bypass_conv.bias)

        self.bypass = nn.Sequential(
            spectral_norm(self.bypass_conv),
            nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_size, padding=0)
        )

    def forward(self, x):
        out_x = self.model(x) + self.bypass(x)

        return out_x


class Encoder(nn.Module):
    def __init__(self, z_dim, ch,size):
        super(Encoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size

        self.enc_block_1 = first_resblock_enc(ch, ch * 2, pooling_size=(1, 2))  # 1, 512 ->  1,256
        self.enc_block_2 = resblock_enc(ch * 2, ch * 4, pooling_size=(1, 2))  # 1, 256 -> 1,128
        self.enc_block_3 = resblock_enc(ch * 4, ch * 8, pooling_size=(1, 2))  # 1,128 -> 1,64
        self.enc_block_4 = resblock_enc(ch * 8, ch * 16, pooling_size=(1, 2))  # 1,128 -> 1, 64
        self.enc_block_5 = resblock_enc(ch * 16, ch * 32, pooling_size=(1, 2))  # 1,64 -> 1,32

        self.blocks = nn.Sequential(
            self.enc_block_1,
            self.enc_block_2,
            self.enc_block_3,
            self.enc_block_4,
            self.enc_block_5,
            nn.ReLU()
        )

        conv_output_size = ch * 32 * int(self.size/2**5)

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mu = nn.Sequential(
            nn.Linear(32, z_dim)
        )

        self.var = nn.Sequential(
            nn.Linear(32, z_dim),
            nn.Softplus()
        )

        # 가중치 초기화
        nn.init.xavier_uniform_(self.fc[0].weight)
        nn.init.xavier_uniform_(self.fc[3].weight)
        nn.init.zeros_(self.fc[0].bias)
        nn.init.zeros_(self.fc[3].bias)
        nn.init.xavier_uniform_(self.mu[0].weight)
        nn.init.zeros_(self.mu[0].bias)
        nn.init.xavier_uniform_(self.var[0].weight)
        nn.init.zeros_(self.var[0].bias)
        # BatchNorm1d 초기화
        nn.init.ones_(self.fc[1].weight)  # 첫 번째 BatchNorm1d의 weight (gamma)
        nn.init.zeros_(self.fc[1].bias)  # 첫 번째 BatchNorm1d의 bias (beta)
        nn.init.ones_(self.fc[4].weight)  # 두 번째 BatchNorm1d의 weight (gamma)
        nn.init.zeros_(self.fc[4].bias)  # 두 번째 BatchNorm1d의 bias (beta)


    def forward(self, x):
        encoded = self.blocks(x)
        encoded = self.fc(encoded.view(-1, encoded.shape[1] * encoded.shape[2] * encoded.shape[3]))
        mu = self.mu(encoded)
        var = self.var(encoded)
        z = reparameterization(mu, var,self.device)
        return z, mu, var


class resblock_dec(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample):
        super(resblock_dec, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.zeros_(self.conv1.bias)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=up_sample, mode='bilinear', align_corners=True),
            self.conv1
        )

        self.bypass_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                     bias=True)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.)
        nn.init.zeros_(self.bypass_conv.bias)

        self.bypass = nn.Sequential(
            nn.Upsample(scale_factor=up_sample, mode='bilinear', align_corners=True),
            self.bypass_conv
        )

    def forward(self, x):
        out_x = self.model(x) + self.bypass(x)
        return out_x


class Decoder(nn.Module):
    def __init__(self, z_dim, ch,size):
        super(Decoder, self).__init__()
        self.ch = ch
        self.size = size

        self.fc = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, ch * 32 * int(size/2**5)),
            nn.BatchNorm1d( ch* 32 * int(size/2**5)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.dec_block_1 = resblock_dec(ch * 32, ch * 16, up_sample=(1, 2))  # 1,16 ->  1,32
        self.dec_block_2 = resblock_dec(ch * 16, ch * 8, up_sample=(1, 2))  # 1,32 ->  1,64
        self.dec_block_3 = resblock_dec(ch * 8, ch * 4, up_sample=(1, 2))  # 1,64 ->  1,128
        self.dec_block_4 = resblock_dec(ch * 4, ch * 2, up_sample=(1, 2))  # 1,128 -> 1,256

        self.blocks = nn.Sequential(
            self.dec_block_1,
            self.dec_block_2,
            self.dec_block_3,
            self.dec_block_4
        )

        self.decoder_mu = nn.Sequential(
            resblock_dec(ch * 2, ch, up_sample=(1, 2)))# 1,256 -> 1,512
        self.decoder_var = nn.Sequential(
            resblock_dec(ch * 2, ch, up_sample=(1, 2)),# 1,256 -> 1,512
            nn.Softplus())
        # 가중치 초기화
        nn.init.xavier_uniform_(self.fc[0].weight)
        nn.init.xavier_uniform_(self.fc[3].weight)
        nn.init.xavier_uniform_(self.fc[6].weight)
        nn.init.zeros_(self.fc[0].bias)
        nn.init.zeros_(self.fc[3].bias)
        nn.init.zeros_(self.fc[6].bias)

        # BatchNorm1d 초기화
        nn.init.ones_(self.fc[1].weight)  # 첫 번째 BatchNorm1d의 weight (gamma)
        nn.init.zeros_(self.fc[1].bias)  # 첫 번째 BatchNorm1d의 bias (beta)

        nn.init.ones_(self.fc[4].weight)  # 두 번째 BatchNorm1d의 weight (gamma)
        nn.init.zeros_(self.fc[4].bias)  # 두 번째 BatchNorm1d의 bias (beta)

        nn.init.ones_(self.fc[7].weight)  # 세 번째 BatchNorm1d의 weight (gamma)
        nn.init.zeros_(self.fc[7].bias)  # 세 번째 BatchNorm1d의 bias (beta)


    def forward(self, z):
        xx = self.fc(z)
        decoded = self.blocks(xx.view(-1, self.ch * 32, 1, int(self.size/2**5)))
        mu = self.decoder_mu(decoded)
        var = self.decoder_var(decoded)
        return mu, var




class VAE(nn.Module):
    def __init__(self, z_dim, ch,size ):
        super(VAE, self).__init__()
        self.enc = Encoder(z_dim, ch,size)
        self.dec = Decoder(z_dim, ch,size)

    def forward(self, x):
        z, mu, var = self.enc(x)  # エンコード
        x_mu, x_var = self.dec(z)  # デコード
        return x_mu, x_var, z, mu, var

    def loss(self, x):
        z, mu, var = self.enc(x)
        x_mu, x_var = self.dec(z)
        KL = gauss_unitgauss_kl(mu, var) * 1
        rec_loss = rec_loss_norm4D(x, x_mu, x_var)
        lower_bound = [-KL, -rec_loss]
        return -sum(lower_bound), KL, rec_loss


    def waveshow(self, true, reconstruction, i_check=[0, 1, 2, 3]):
        rec_x_x = reconstruction
        n_fig = len(i_check)
        fig, ax = plt.subplots(n_fig, 1, figsize=(10, 8), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.1, bottom=0.07, top=0.9, right=0.97, wspace=0.25, hspace=0.2)
        for i in range(n_fig):
            ax[i].plot(true[i, -1, 0, :], label='True', color='r')
            ax[i].plot(true[i, -2, 0, :], label='True', color='r',linestyle=':')
            ax[i].plot(rec_x_x[i, -1, 0, :], label='rec_x_x',color='C0')
            ax[i].plot(rec_x_x[i, -2, 0, :], label='rec_x_x',color='C0',linestyle=':')
            ax[i].legend(loc='upper right')
            ax[i].grid(alpha=0.5)
        plt.show()
