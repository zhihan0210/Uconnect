import numpy as np
import matplotlib.pyplot as plt
from image2 import ParamImg
from projectors import ParamProj, forward_projection, back_projection
from recoCT import ParamRecon, ParamRecon_dTV, recoCT_PWLS, recoCT_dTV
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

#x_init = np.zeros(paramImg.img_shape)
#y_prior = np.zeros(y_all[0].shape)
#for i_e in range(len(paramImg.energy_tgt)):
#    y_prior = y_prior + y_all[i_e]

#paramRecon_prior = ParamRecon(sum(I_all), 50, bckg=0, prior='Huber', beta=400, delta=0.00001, nInnerIter=25, theta=1,
#                                eta=0.7, eps=0.00001).get_paramRecon()
#x_prior = recoCT_PWLS(x_init, y_prior, paramImg, proj_id, paramRecon_prior)
#np.save("/home2/zwang/recon/results/dTV/x_prior.npy",x_prior)
#exit()

x_prior = np.load("/home2/zwang/recon/results/dTV/x_prior.npy")

beta_all = [0.75, 3.75, 7.5, 15.0, 22.5, 30.0, 37.5, 45.0, 52.5, 60.0]

for beta in beta_all:
    paramRecon = ParamRecon_dTV(I_all[0], 1000, bckg=0, prior='dTV', beta=beta, delta=0.0001, nInnerIter=25, theta=1, eta=0.7, eps=0.0001).get_paramRecon()
    x = recoCT_dTV(y_all[0], x_prior, paramImg, proj_id, paramRecon)
    name = '/home2/zwang/recon/results/dTV/beta'+str(beta)+'_40keV.npy'
    np.save(name, x)


