import numpy as np
import tensorflow as tf
import copy
from imagePrior import imagePrior
from scipy.optimize import fmin_l_bfgs_b
from recoCT import compute_grad, compute_grad_back, computeP
from model import model

tf.config.experimental.enable_tensor_float_32_execution(False)

class ParamReg:
    def __init__(self, alpha, gamma, paramRecon):
        self.alpha = alpha
        self.gamma = gamma
        self.paramRecon = paramRecon

    def get_paramReg(self):
        return self

# compute sum(0.5*gamma_k*||f_k(z)-s_k*x_k||²) and its gradient
def Ls_CNN(im,y,list_s,paramImg,paramReg):
    input_im = tf.convert_to_tensor(im.reshape(paramImg.tensor_shape))
    f = 0
    for i_e in range(len(paramImg.energy_tgt)):
        f_temp = 0.5*paramReg.gamma[i_e]*tf.square(model(input_im, i_e, paramImg)-list_s[i_e]*y[i_e])
        f_temp = f_temp.numpy().reshape(paramImg.img_shape)
        f = f + np.sum(f_temp)
    ff = []
    g = np.zeros(paramImg.img_shape)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_im)
        for i_e in range(len(paramImg.energy_tgt)):
            ff.append(0.5*paramReg.gamma[i_e]*tf.square(model(input_im, i_e, paramImg) - list_s[i_e] * y[i_e]))

    for i_e in range(len(paramImg.energy_tgt)):
        g_temp = tape.gradient(ff[i_e], input_im)
        g = g + g_temp.numpy().reshape(paramImg.img_shape)
    g = g.flatten(order='F')

    return f, g

# compute Ls_CNN + alpha*H(z) and its gradient
def fullcostsum_vec(im_vec,y,list_s,paramImg,paramReg):
    im = im_vec.reshape(paramImg.img_shape, order='F')
    f, g = Ls_CNN(im, y, list_s, paramImg, paramReg)
    fr, gr = imagePrior(im, paramReg.paramRecon)

    f = f + paramReg.alpha * fr
    g = g + paramReg.alpha * gr

    return f, g

# minimize sum(0.5*gamma_k*||f_k(z)-s_k*x_k||²) + alpha*H(z) with respect to z
def min_z_lbfgs(ct_input, ct_tgt, list_s, paramImg, paramReg):
    z = ct_input
    z_vecinit = z.flatten(order='F')

    bnds = [(0, 0.5)] * z_vecinit.size   # set arbitrarily to define the bounds for z

    cost_1var = lambda im_vec: fullcostsum_vec(im_vec, ct_tgt, list_s, paramImg, paramReg)

    z_vec, y_opt, opt_info = fmin_l_bfgs_b(cost_1var, z_vecinit, fprime=None, bounds=bnds, m=9, factr=10, pgtol=1e-07,
                                           iprint=99, maxiter=paramReg.paramRecon.nb_iter)
    z = z_vec.reshape(paramImg.img_shape, order='F')

    z = z.reshape(paramImg.tensor_shape, order='F')

    return z

