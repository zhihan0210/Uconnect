import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import time
from scipy.optimize import fmin_l_bfgs_b
from projectors import ParamProj, forward_projection, back_projection
from recoCT import ParamRecon, ParamRecon_dTV, recoCT_LBFGSB, fullcost, recoCT_PWLS, recoCT_PWLS_z, recoCT_dTV, recoCT_JTV
from regularization import ParamReg, min_z_lbfgs
from image_MH_normalized import ParamImg, resize_img
from MHUconnect import MHUconnect_method
import astra

paramImg = ParamImg().get_paramImg()
paramProj = ParamProj(n_angles=120, vox_size=0.75, beam_type='fanflat', det_width=1.2, det_count=750, source_origin=600, origin_det=600, proj_type='line_fanflat').get_paramProj()
# paramProj = ParamProj(n_angles=120, vox_size=3, beam_type='fanflat', det_width=1.2, det_count=750, source_origin=600, origin_det=600, proj_type='line_fanflat').get_paramProj()

vol_geom = astra.creators.create_vol_geom(paramImg.img_shape)
proj_geom = astra.create_proj_geom(paramProj.beam_type, round(paramProj.det_width/paramProj.vox_size, 2), paramProj.det_count, np.linspace(0, 2 * np.pi, paramProj.n_angles, endpoint=False), round(paramProj.source_origin/paramProj.vox_size), round(paramProj.origin_det/paramProj.vox_size))
proj_id = astra.create_projector(paramProj.proj_type, proj_geom, vol_geom)

I_all = [200000/40 for i_e in range(len(paramImg.energy_tgt))]

array_y_all = np.load("/home/zhihan/recon_results/xcat/y_all.npy")

y_all = []
for i_e in range(len(paramImg.energy_tgt)):
    y_all.append(array_y_all[i_e,:,:])

# poids
sum_y_select = []
for i_e in range(len(paramImg.energy_tgt)):
    sort = np.sort(y_all[i_e].flatten())
    select = sort[0:int(np.size(sort) / 2)]
    sum_y_select.append(np.sum(select))
gamma_all = [i/np.sum(sum_y_select) for i in sum_y_select]


beta_all = [4500000,5000000]

for beta in beta_all:
    paramRecon = ParamRecon(I_all, 50, bckg=0, prior='Huber', beta=beta, delta=0.00001).get_paramRecon()  # vary beta 25
    paramRecon_Reg = ParamRecon(I_all, 300, bckg=0, prior='Huber', beta=0, delta=0.00001)  # no use: beta

    # paramReg = ParamReg(alpha=0.1, gamma=gamma_all,
    #                    paramRecon=paramRecon_Reg).get_paramReg()  # dTV: 0.0005, Huber: 0.01 xcat128
    paramReg = ParamReg(alpha=0.07, gamma=gamma_all,
                        paramRecon=paramRecon_Reg).get_paramReg()  # dTV: 0.0005, Huber: 0.01
    x = MHUconnect_method(y_all, paramImg, proj_id, paramRecon, paramReg)
    name = '/home/zhihan/recon_results/xcat/MHUconnect/beta' + str(beta) + '_alpha007_gamma2.npy'
    np.save(name, x)
