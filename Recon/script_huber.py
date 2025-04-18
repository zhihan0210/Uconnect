import numpy as np
import matplotlib.pyplot as plt
import time
from projectors import ParamProj, forward_projection, back_projection
from recoCT import ParamRecon, recoCT_LBFGSB, recoCT_PWLS
from image import ParamImg
import astra

paramImg = ParamImg().get_paramImg()
paramProj = ParamProj(n_angles=120, vox_size=3, beam_type='fanflat', det_width=1.2, det_count=750, source_origin=600, origin_det=600, proj_type='line_fanflat').get_paramProj()

vol_geom = astra.creators.create_vol_geom(paramImg.img_shape)
proj_geom = astra.create_proj_geom(paramProj.beam_type, round(paramProj.det_width/paramProj.vox_size, 2), paramProj.det_count, np.linspace(0, 2 * np.pi, paramProj.n_angles, endpoint=False), round(paramProj.source_origin/paramProj.vox_size), round(paramProj.origin_det/paramProj.vox_size))
proj_id = astra.create_projector(paramProj.proj_type, proj_geom, vol_geom)

I_all = [200000/40 for i_e in range(len(paramImg.energy_tgt))]

array_y_all = np.load("/home2/zwang/recon/results/y_all.npy")

y_all = []
for i_e in range(len(paramImg.energy_tgt)):
    y_all.append(array_y_all[i_e,:,:])


x_init = np.zeros(paramImg.img_shape)

beta_all = [500000]
for beta in beta_all:
    paramRecon = ParamRecon(I_all[0], 1000, bckg=0, prior='Huber', beta=beta,
                            delta=0.0001).get_paramRecon()  # vary beta
    x = recoCT_PWLS(x_init, y_all[0], paramImg, proj_id, paramRecon)
    name = '/home2/zwang/recon/results/huber/beta'+str(beta)+'_40keV.npy'
    np.save(name, x)
