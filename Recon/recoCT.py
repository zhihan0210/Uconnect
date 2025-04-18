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

class ParamRecon_JTV(ParamRecon):
    def __init__(self, I, nb_iter, bckg, prior, beta, delta, nInnerIter, theta):
        ParamRecon.__init__(self, I, nb_iter, bckg, prior, beta, delta)
        self.nInnerIter = nInnerIter
        self.theta = theta

class ParamRecon_dTV(ParamRecon):
    def __init__(self, I, nb_iter, bckg, prior, beta, delta, nInnerIter, theta, eta, eps):
        ParamRecon.__init__(self, I, nb_iter, bckg, prior, beta, delta)
        self.nInnerIter = nInnerIter
        self.theta = theta
        self.eta = eta
        self.eps = eps


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

# conventional reconstruction by PWLS (Penalized Weighted Least Squares) algorithm: minimize 0.5*||Ax-b||²_w + beta*R(x) without the class paramImg
def recoCT_PWLS_simple(x_init, y, img_shape, proj_id, paramRecon):
    diff = y - paramRecon.bckg
    l = np.where(diff > 0, np.log(paramRecon.I / diff), 0)

    w = np.where(diff > 0, np.square(diff) / y, 0)

    N = img_shape[0]
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
        # x = (D * x_rec) / D
        # x = np.where(x > 0, x, 0)
        x = np.where(x_rec > 0, x_rec, 0)
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

def recoCT_PWLS_z_normalization(x_init, y, fz, beta, paramImg, proj_id, paramRecon):
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
        x = (D * x_rec + beta * fz) / (D + beta)
        x = np.where(x > 0, x, 0)

    return x

def diff_back2D(y):
    l = np.array([0, 1, -1])
    vect = np.zeros((9, 2))
    ii = 0
    for i in range(3):
        for j in range(3):
            vect[ii, :] = np.array([l[i], l[j]])
            ii = ii + 1
    vect = np.delete(vect, 0, axis=0)
    im = np.zeros([np.shape(y)[0], np.shape(y)[1]])
    for i in range(np.size(vect, 0)):
        shift_vect = vect[i,:].astype(int)
        im_shifted = np.roll(np.roll(y[:, :, i], -shift_vect[0], axis=0), -shift_vect[1], axis=1)
        im = im + (y[:, :, i] - im_shifted) / np.linalg.norm(shift_vect)
    x = im
    return x

def compute_grad(x):
    Dx = np.zeros([np.shape(x)[0], np.shape(x)[1], 2])
    Dx[:, :, 0] = x - np.roll(np.roll(x, 1, axis=0), 0, axis=1)
    Dx[:, :, 1] = x - np.roll(np.roll(x, 0, axis=0), 1, axis=1)
    return Dx

def compute_grad_back(y):
    x = np.zeros([np.shape(y)[0], np.shape(y)[1]])
    x = x + y[:, :, 0] - np.roll(np.roll(y[:, :, 0], -1, axis=0), 0, axis=1)
    x = x + y[:, :, 1] - np.roll(np.roll(y[:, :, 1], 0, axis=0), -1, axis=1)
    return x

def computeP(u, xi):
    res = np.zeros(np.shape(u))
    res[:,:,0] = u[:,:,0] - u[:,:,0] * (xi[:,:,0]**2) - xi[:,:,0]*xi[:,:,1]*u[:,:,1]
    res[:,:,1] = u[:,:,1] - u[:,:,1] * (xi[:,:,1]**2) - xi[:,:,0]*xi[:,:,1]*u[:,:,0]
    return res


def recoCT_JTV(y_all, paramImg, proj_id, paramRecon_JTV):
    #assert paramRecon_JTV.prior == 'JTV', "paramRecon.prior should be JTV"

    b = []
    w = []

    for i_e in range(len(paramImg.energy_tgt)):
        diff = y_all[i_e] - paramRecon_JTV.bckg
        b.append(np.where(diff > 0, np.log(paramRecon_JTV.I[i_e] / diff), 0))
        w.append(np.where(diff > 0, np.square(diff) / y_all[i_e], 0))

    N = paramImg.img_shape[0]

    theta = paramRecon_JTV.theta
    scale = 6
    sum_y_select = []
    for i_e in range(len(paramImg.energy_tgt)):
        sort = np.sort(y_all[i_e].flatten())
        select = sort[0:int(np.size(sort) / 4)]
        sum_y_select.append(np.sum(select))
        w[i_e] = scale * w[i_e] / np.sum(select)

    beta = scale * paramRecon_JTV.beta / np.max(sum_y_select)
    x = np.random.rand(N, N)

    for k in range(100):
        if (paramRecon_JTV.prior == 'JTV1'):
            x = diff_back2D(diff_operator2D(x))
            normalisation = np.sqrt(np.sum(x ** 2))
            x = x / normalisation
            s = np.sqrt(np.sum(diff_operator2D(x) ** 2))
        elif (paramRecon_JTV.prior == 'JTV2'):
            x = compute_grad_back(compute_grad(x))
            normalisation = np.sqrt(np.sum(x ** 2))
            x = x / normalisation
            s = np.sqrt(np.sum(compute_grad(x) ** 2))
    L = s

    sigma = 0.8 * 1 / L
    tau = 0.8 * 1 / L

    x = [np.zeros([N, N]) for i_e in range(len(paramImg.energy_tgt))]

    xbar = copy.deepcopy(x)

    if (paramRecon_JTV.prior == 'JTV1'):
        z = [np.zeros(np.shape(diff_operator2D(xbar[i_e]))) for i_e in range(len(paramImg.energy_tgt))]
    elif (paramRecon_JTV.prior == 'JTV2'):
        z = [np.zeros(np.shape(compute_grad(xbar[i_e]))) for i_e in range(len(paramImg.energy_tgt))]

    D_rec = []
    for i_e in range(len(paramImg.energy_tgt)):
        D_rec.append(back_projection(w[i_e] * forward_projection(np.ones([N, N]), proj_id), proj_id))

    for i in range(paramRecon_JTV.nb_iter):
        print("Iteration %(i)d" % {"i": i})
        start = time.time()

        z_temp = []
        nor = 0
        if (paramRecon_JTV.prior == 'JTV1'):
            for i_e in range(len(paramImg.energy_tgt)):
                temp = z[i_e] + sigma * diff_operator2D(xbar[i_e])
                z_temp.append(temp)
                nor = nor + temp ** 2
            normZ = np.sqrt(nor)
        elif (paramRecon_JTV.prior == 'JTV2'):
            for i_e in range(len(paramImg.energy_tgt)):
                temp = z[i_e] + sigma * compute_grad(xbar[i_e])
                z_temp.append(temp)
                #nor = nor + temp ** 2
                nor = nor + temp[:, :, 0] ** 2 + temp[:, :, 1] ** 2
            #normZ = np.sqrt(nor)
            normZ = np.zeros(np.shape(temp))
            normZ[:, :, 0] = np.sqrt(nor)
            normZ[:, :, 1] = np.sqrt(nor)
        for i_e in range(len(paramImg.energy_tgt)):
            temp = np.where(normZ>beta,beta,normZ) * (z_temp[i_e] / normZ)
            z[i_e] = np.where(np.isnan(temp),0,temp)
        x_old = copy.deepcopy(x)
        x_temp = [[] for i_e in range(len(paramImg.energy_tgt))]
        if (paramRecon_JTV.prior == 'JTV1'):
            for i_e in range(len(paramImg.energy_tgt)):
                x_temp[i_e] = x_old[i_e] - tau * diff_back2D(z[i_e])
        elif (paramRecon_JTV.prior == 'JTV2'):
            for i_e in range(len(paramImg.energy_tgt)):
                x_temp[i_e] = x_old[i_e] - tau * compute_grad_back(z[i_e])

        for in_k in range(paramRecon_JTV.nInnerIter):
            x_rec = [[] for i_e in range(len(paramImg.energy_tgt))]
            for i_e in range(len(paramImg.energy_tgt)):
                x_rec[i_e] = x[i_e] - back_projection(w[i_e]*(forward_projection(x[i_e],proj_id)-b[i_e]), proj_id) /D_rec[i_e]
                x[i_e] = (D_rec[i_e] * x_rec[i_e] + x_temp[i_e] / tau) / (D_rec[i_e] + 1 / tau)
                x[i_e] = np.where(x[i_e] > 0, x[i_e], 0)

        for i_e in range(len(paramImg.energy_tgt)):
            print(i)
            print(np.sum(x[i_e] - x_old[i_e]))
            xbar[i_e] = x[i_e] + theta * (x[i_e] - x_old[i_e])

        end = time.time()
        print(end - start)
    return x


def recoCT_dTV(y, x_prior, paramImg, proj_id, paramRecon_dTV):
    assert paramRecon_dTV.prior == 'dTV', "paramRecon.prior should be dTV"

    diff = y - paramRecon_dTV.bckg
    b = np.where(diff > 0, np.log(paramRecon_dTV.I / diff), 0)
    w = np.where(diff > 0, np.square(diff) / y, 0)

    N = paramImg.img_shape[0]

    beta = paramRecon_dTV.beta
    theta = paramRecon_dTV.theta

    w = [i / paramRecon_dTV.I for i in w]
    beta = beta / paramRecon_dTV.I

    # compute directional information xi from x_prior
    grad_x_prior = compute_grad(x_prior)
    norm_grad_x_prior_eps = np.sqrt(grad_x_prior[:,:,0]**2 + grad_x_prior[:,:,1]**2 + paramRecon_dTV.eps**2)
    eta = paramRecon_dTV.eta
    xi = np.zeros(np.shape(grad_x_prior))
    xi[:, :, 0] = eta * grad_x_prior[:, :, 0] / norm_grad_x_prior_eps
    xi[:, :, 1] = eta * grad_x_prior[:, :, 1] / norm_grad_x_prior_eps

    x = np.random.rand(N, N)

    for k in range(100):
        x = compute_grad_back(computeP(computeP(compute_grad(x),xi),xi))
        normalisation = np.sqrt(np.sum(x ** 2))
        x = x / normalisation
        s = np.sqrt(np.sum(computeP(compute_grad(x),xi)**2))

    L = s

    sigma = 0.8 * 1 / L
    tau = 0.8 * 1 / L

    x = np.zeros([N, N])

    xbar = copy.deepcopy(x)

    z = np.zeros(np.shape(computeP(compute_grad(xbar),xi)))

    D_rec = back_projection(w * forward_projection(np.ones([N, N]), proj_id), proj_id)

    for i in range(paramRecon_dTV.nb_iter):
        z_temp = z + sigma * computeP(compute_grad(xbar),xi)
        normZ = np.sqrt(z_temp[:,:,0]**2 + z_temp[:,:,1]**2)
        z = np.zeros(np.shape(z_temp))
        z[:, :, 0] = np.where(normZ > beta, beta, normZ) * (z_temp[:, :, 0] / normZ)
        z[:, :, 1] = np.where(normZ > beta, beta, normZ) * (z_temp[:, :, 1] / normZ)
        z = np.where(np.isnan(z), 0, z)

        x_old = copy.deepcopy(x)
        x_temp = x_old - tau * compute_grad_back(computeP(z,xi))

        for inner_k in range(paramRecon_dTV.nInnerIter):
            x_rec = x - back_projection(w * (forward_projection(x, proj_id) - b), proj_id) / D_rec
            x = (D_rec * x_rec + x_temp / tau) / (D_rec + 1 / tau)
            x = np.where(x > 0, x, 0)

        xbar = x + theta * (x - x_old)

    return x
