import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import time
from scipy.optimize import fmin_l_bfgs_b
from projectors import ParamProj, forward_projection, back_projection
from recoCT import ParamRecon, ParamRecon_dTV, recoCT_LBFGSB, fullcost, recoCT_PWLS, recoCT_PWLS_z, recoCT_dTV, recoCT_JTV
from regularization import ParamReg, min_z_lbfgs
from image import ParamImg, resize_img
from Uconnect import Uconnect_method
import astra

paramProj = ParamProj(n_angles=120, vox_size=0.75, beam_type='fanflat', det_width=1.2, det_count=750, source_origin=600, origin_det=600, proj_type='line_fanflat').get_paramProj()

vol_geom = astra.creators.create_vol_geom(paramImg.img_shape)
# the unit of attribute for the function is per pixel
proj_geom = astra.create_proj_geom(paramProj.beam_type, round(paramProj.det_width/paramProj.vox_size, 2), paramProj.det_count, np.linspace(0, 2 * np.pi, paramProj.n_angles, endpoint=False), round(paramProj.source_origin/paramProj.vox_size), round(paramProj.origin_det/paramProj.vox_size))
proj_id = astra.create_projector(paramProj.proj_type, proj_geom, vol_geom)

I_all = [200000/40 for i_e in range(len(paramImg.energy_tgt))]

# define y_k by the Beer-Lambert law as ybar_k=I_k*exp(-Ax_k) and y_k~Poisson(y_k(x_k))
#array_gt_tgt = np.load("dataset/reconstruction/gt_tgt.npy")
#proj_all = []
#for i_e in range(len(paramImg.energy_tgt)):
#    proj_all.append(forward_projection(array_gt_tgt[i_e,:,:].reshape(paramImg.img_shape), proj_id))
#y_all = []
#for i_e in range(len(paramImg.energy_tgt)):
#    ybar = I_all[i_e] * np.exp(-proj_all[i_e])
#    y_all.append(np.random.poisson(ybar))

#np.save("results/proj_all.npy",proj_all)
#np.save("results/y_all.npy",y_all)


array_y_all = np.load("results/y_all.npy")

y_all = []
for i_e in range(len(paramImg.energy_tgt)):
    y_all.append(array_y_all[i_e,:,:])


#gamma_k is proportional to the sum of the first 25 percent of y_k in order to give more weight to the energy bins with more counts
sum_y_select = []
for i_e in range(len(paramImg.energy_tgt)):
    sort = np.sort(y_all[i_e].flatten())
    select = sort[0:int(np.size(sort) / 4)]
    sum_y_select.append(np.sum(select))
gamma_all = [i/np.sum(sum_y_select) for i in sum_y_select]




beta_all = [5000000, 6000000, 7000000]
for beta in beta_all:
    paramRecon = ParamRecon(I_all, 50, bckg=0, prior='Huber', beta=beta, delta=0.00001).get_paramRecon()  # vary beta
    paramRecon_Reg = ParamRecon(I_all, 300, bckg=0, prior='Huber', beta=0, delta=0.00001)  # no use: beta

    paramReg = ParamReg(alpha=1, gamma=gamma_all,
                        paramRecon=paramRecon_Reg).get_paramReg()
    x = Uconnect_method(y_all, paramImg, proj_id, paramRecon, paramReg)
    name = 'results/Uconnect/beta'+str(beta)+'_alpha1.npy'
    np.save(name, x)
