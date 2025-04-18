import numpy as np
from denoisingCT import ParamDenoising_JTV, denoise_JTV

array_x_hat_all = np.load("/home2/zwang/denoise/results/x_hat.npy")

x_noisy_all = []
for i_e in range(6):
    x_noisy_all.append(array_x_hat_all[i_e, :, :])

beta_all = [0.002, 0.005, 0.003, 0.004, 0.006, 0.007, 0.008, 0.009]

for beta in beta_all:
    paramDenoising_JTV = ParamDenoising_JTV(prior='JTV2', nb_iter = 1000, beta=beta, delta=0.00001, theta=1).get_paramDenoising()
    x_all = denoise_JTV(x_noisy_all, paramDenoising_JTV)
    name = '/home2/zwang/denoise/results/JTV/beta' + str(beta) + '.npy'
    np.save(name, x_all)
