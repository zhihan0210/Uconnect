import numpy as np
import matplotlib.pyplot as plt
from image import ParamImg
from regularization import ParamReg
from denoisingCT import ParamDenoising, denoising_Uconnect

paramImg = ParamImg().get_paramImg()

array_x_hat_all = np.load("/home/zhihan/recon_results/real_data/x_hat.npy")

x_hat_all = []
for i_e in range(len(paramImg.energy_tgt)):
    x_hat_all.append(array_x_hat_all[i_e,:,:])

gamma_all = np.load("/home/zhihan/recon_results/real_data/gamma_all.npy").tolist()

#beta_all = [10,15,20,25,30,40,50,60]
beta_all = [60]
for beta in beta_all:
    paramDenoising = ParamDenoising(prior='Huber', nb_iter = 1, beta=beta, delta=0.00001).get_paramDenoising()
    paramDenoising_Reg = ParamDenoising(prior='Huber', nb_iter = 300, beta=0, delta=0.00001)  # no use: beta

    paramReg = ParamReg(alpha=0.4, gamma=gamma_all,
                        paramRecon=paramDenoising_Reg).get_paramReg()  # dTV: 0.0005, Huber: 0.01
    x = denoising_Uconnect(x_hat_all, paramImg, paramReg, paramDenoising)
    name = '/home/zhihan/recon_results/real_data/Uconnect/allunet/beta'+str(beta)+'_alpha'+str(paramReg.alpha)+'.npy'
    np.save(name, x)
