import numpy as np
import matplotlib.pyplot as plt
from recoCT import recoCT_PWLS_z_normalization
from regularization import min_z_lbfgs_MH_normalized
import astra
import copy

def MHUconnect_method(y_all, paramImg, proj_id, paramRecon, paramReg):
    x_all = []
    array_x_all = np.load("/home/zhihan/data/recon_results/xcat/pre_x_all.npy")
    for i_e in range(len(paramImg.energy_tgt)):
        x_all.append(array_x_all[i_e, :, :].reshape(paramImg.tensor_shape))
    z = np.load('/home/zhihan/data/recon_results/xcat/MHUconnect/initz_alpha'+str(paramReg.alpha)+'.npy')
    # prev_psnrs = [0] * len(paramImg.energy_tgt)
    for i in range(50):
        z = min_z_lbfgs_MH_normalized(z, x_all, paramImg, paramReg)
        np.save('/home/zhihan/data/recon_results/xcat/MHUconnect/z/alpha' + str(paramReg.alpha) + '_beta' + str(
                paramRecon.beta) + '_' + str(i) + '.npy', z)
        output = paramImg.model(z)

        for i_e in range(len(paramImg.energy_tgt)):
            fz = output[i_e].numpy().reshape(paramImg.img_shape)
            mu_max = paramImg.mu_max[i_e]
            pRecon = copy.deepcopy(paramRecon)
            pRecon.I = paramRecon.I[i_e]
            x_all[i_e] = (recoCT_PWLS_z_normalization(x_all[i_e].reshape(paramImg.img_shape), y_all[i_e], mu_max*fz, pRecon.beta*paramReg.gamma[i_e], paramImg, proj_id, pRecon)).reshape(paramImg.tensor_shape)
        np.save('/home/zhihan/data/recon_results/xcat/MHUconnect/x/alpha' + str(
                paramReg.alpha) + '_beta' + str(paramRecon.beta) + '_' + str(i) + '.npy', x_all)
        print('finish ' + str(i) + '/50 epochs')
    return x_all
