import numpy as np
from denoisingCT import ParamDenoising, denoise_lbfgs

array_x_hat_all = np.load("/home/zhihan/recon_results/real_data/x_hat.npy")

x_noisy_all = []
for i_e in range(6):
    x_noisy_all.append(array_x_hat_all[i_e, :, :])

beta_all = [13]

for beta in beta_all:
    paramDenoising = ParamDenoising(prior='Huber', nb_iter = 1000, beta=beta, delta=0.00001).get_paramDenoising()

    x_all = []
    for i_e in range(6):
        x = denoise_lbfgs(x_noisy_all[i_e], paramDenoising)
        x_all.append(x)
    name = '/home/zhihan/recon_results/real_data/Huber/beta' + str(beta) + '.npy'
    np.save(name, x_all)