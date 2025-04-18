import numpy as np
from denoisingCT import ParamDenoising_dTV, denoise_dTV

array_x_hat_all = np.load("/home2/zwang/denoise/results/x_hat.npy")

x_noisy_all = []
for i_e in range(6):
    x_noisy_all.append(array_x_hat_all[i_e, :, :])

x_prior = np.load("/home2/zwang/denoise/results/dTV/x_prior.npy")

beta_all = [0.02,0.03,0.04,0.05,0.004,0.003,0.002]

for beta in beta_all:
    paramDenoising_dTV = ParamDenoising_dTV(prior='dTV', nb_iter = 1000, beta=beta, delta=0.00001, theta=1, eta=0.7, eps=0.00001).get_paramDenoising()
    x_all = []
    for i_e in range(6):
        x = denoise_dTV(x_noisy_all[i_e], x_prior, paramDenoising_dTV)
        x_all.append(x)
    name = '/home2/zwang/denoise/results/dTV/beta' + str(beta) + '.npy'
    np.save(name, x_all)
