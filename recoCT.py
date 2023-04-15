import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from projectors import forward_projection, back_projection
from imagePrior import imagePrior, imagePrior_SPS
import matplotlib.pyplot as plt
import copy
import time


class ParamRecon:
    def __init__(self, I, nb_iter, bckg, prior, beta, delta):
        self.I = I # intensity
        self.nb_iter = nb_iter
        self.bckg = bckg # background
        self.prior = prior
        self.beta = beta
        self.delta = delta

    def get_paramRecon(self):
        return self


def log_lik(xvect, y, paramImg, proj_id, paramRecon):
    N = paramImg.img_shape[0]
    x = xvect.reshape((N, N), order='F')
    exp_attn = np.exp(- forward_projection(x, proj_id))
    ybar = paramRecon.I * exp_attn + paramRecon.bckg

    f = y * np.log(ybar) - ybar
    f = - np.sum(f)

    ratio = y / ybar
    g = back_projection(paramRecon.I * exp_attn * (1 - ratio), proj_id)
    g = -(g.flatten(order='F'))

    return f, g


def fullcost(xvect, y, paramImg, proj_id, paramRecon):
    N = paramImg.img_shape[0]

    fd, gd = log_lik(xvect, y, paramImg, proj_id, paramRecon)

    x = xvect.reshape((N, N), order='F')
    fr, gr = imagePrior(x, paramRecon)

    f = fd + paramRecon.beta * fr
    g = gd + paramRecon.beta * gr

    return f, g

# conventional reconstruction by L-BFGS algorithm: minimize 0.5*||Ax-b||² + beta*R(x) with b = log(I/y)
def recoCT_LBFGSB(x_init, y, paramImg, proj_id, paramRecon):
    N = paramImg.img_shape[0]
    xvect_init = x_init.flatten(order='F')
    bnds = [(0, 0.3)] * xvect_init.size   # set arbitrarily to define the bounds (min, max) pairs for each element in x. It needs to be changed according to different data 
    fullcost_1var = lambda xvect: fullcost(xvect, y, paramImg, proj_id, paramRecon)
    start = time.time()
    xvect, y_opt, opt_info = fmin_l_bfgs_b(fullcost_1var, xvect_init, fprime=None, bounds=bnds, m=9, factr=1e7, iprint=99, maxiter=paramRecon.nb_iter)
    end = time.time()
    print(end - start)
    x = xvect.reshape((N, N), order='F')
    return x


# conventional reconstruction by PWLS (Penalized Weighted Least Squares) algorithm: minimize 0.5*||Ax-b||²_w + beta*R(x)
def recoCT_PWLS(x_init, y, paramImg, proj_id, paramRecon):
    diff = y - paramRecon.bckg
    l = np.where(diff > 0, np.log(paramRecon.I / diff), 0)

    w = np.where(diff > 0, np.square(diff) / y, 0)

    N = paramImg.img_shape[0]
    img_one = np.ones([N, N])
    D = back_projection(w * forward_projection(img_one, proj_id), proj_id)

    x = x_init
    for i in range(paramRecon.nb_iter):
        print("Iteration %(i)d" % {"i": i})
        start = time.time()
        grad = back_projection(w * (forward_projection(x, proj_id) - l), proj_id)
        end = time.time()
        print(end - start)
        x_rec = np.where(D == 0, 0, x - grad / D)
        x_reg, omega_reg = imagePrior_SPS(x, paramRecon)
        x = (D * x_rec + 4 * paramRecon.beta * omega_reg * x_reg) / (D + 4 * paramRecon.beta * omega_reg)
        x = np.where(x > 0, x, 0)
    return x


# conventional reconstruction by PWLS (Penalized Weighted Least Squares) algorithm given z: minimize 0.5*||Ax_k-b_k||²_w + 0.5*beta*||f_k(z)-x_k||²
def recoCT_PWLS_z(x_init, y, fz, beta, s, paramImg, proj_id, paramRecon):
    diff = y - paramRecon.bckg
    l = np.where(diff > 0, np.log(paramRecon.I / diff), 0)

    w = np.where(diff > 0, np.square(diff) / y, 0)

    N = paramImg.img_shape[0]
    img_one = np.ones([N, N])
    D = back_projection(w * forward_projection(img_one, proj_id), proj_id)

    x = x_init
    for i in range(paramRecon.nb_iter):
        print("Iteration %(i)d" % {"i": i})
        start = time.time()
        grad = back_projection(w * (forward_projection(x, proj_id) - l), proj_id)
        end = time.time()
        print(end - start)
        x_rec = np.where(D == 0, 0, x - grad / D)
        x = (D * x_rec + beta * s * fz) / (D + beta * np.square(s))
        x = np.where(x > 0, x, 0)

    return x



