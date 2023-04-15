import numpy as np
import matplotlib.pyplot as plt
from recoCT import ParamRecon, recoCT_PWLS, recoCT_PWLS_z
from projectors import ParamProj, forward_projection, back_projection
from regularization import min_z_lbfgs
import astra
import copy
from model import model

def Uconnect_method(y_all, paramImg, proj_id, paramRecon, paramReg):
    x_all = []
    
    # initialize x_all by PWLS without penalty
    #x_init = np.zeros(paramImg.img_shape)
    #for i_e in range(len(paramImg.energy_tgt)):
    #    param_preRecon = ParamRecon(I_all[i_e], 1000, bckg=0, prior='Huber', beta=0, delta=0.00001).get_paramRecon()
    #    x_temp = recoCT_PWLS(x_init, y_all[i_e], paramImg, proj_id, param_preRecon)
    #    x_all.append(x_temp)
    #np.save("results/pre_x_all.npy", x_all)
    array_x_all = np.load("results/pre_x_all.npy")
    for i_e in range(len(paramImg.energy_tgt)):
        x_all.append(array_x_all[i_e, :, :].reshape(paramImg.tensor_shape))
       
    # scale factor for each energy
    #s_hat_all = []
    #for i_e in range(len(paramImg.energy_tgt)):
    #    s_hat_all.append(np.sum(x_all[0]) / np.sum(x_all[i_e]))
    #np.save("results/s_hat_all.npy", s_hat_all)
    s_hat_all = np.load("results/s_hat_all.npy").tolist()
    
    # initialize z for early convergence
    #z = 0.01 * np.ones(paramImg.tensor_shape)
    #paramRecon_preReg = ParamRecon(I_all,300,bckg=0,prior='Huber',beta=0,delta=0.00001)
    #param_preReg = ParamReg(alpha=paramReg.alpha, gamma=gamma_all, paramRecon=paramRecon_preReg).get_paramReg()
    #z = min_z_lbfgs(z, x_all, s_hat_all, paramImg, param_preReg)
    #np.save("results/Uconnect/initz_alpha1.npy",z)
    z = np.load("results/Uconnect/initz_alpha1.npy")

    for i in range(50):
    	# minimize for z
        z = min_z_lbfgs(z, x_all, s_hat_all, paramImg, paramReg)
        np.save('results/Uconnect/z/alpha' + str(paramReg.alpha) + '_beta' + str(
                paramRecon.beta) + '_' + str(i) + '.npy', z)
        # reconstruct each x_k given z
        for i_e in range(len(paramImg.energy_tgt)):
            fz = model(z, i_e, paramImg).numpy().reshape(paramImg.img_shape)
            s = s_hat_all[i_e]
            pRecon = copy.deepcopy(paramRecon)
            pRecon.I = paramRecon.I[i_e]
            x_all[i_e] = (recoCT_PWLS_z(x_all[i_e].reshape(paramImg.img_shape), y_all[i_e], fz, pRecon.beta*paramReg.gamma[i_e], s, paramImg, proj_id, pRecon)).reshape(paramImg.tensor_shape)
        print('finish '+str(i) + '/50 epochs')
    return x_all
