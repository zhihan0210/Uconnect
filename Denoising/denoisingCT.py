import numpy as np
import copy
from imagePrior import imagePrior
from regularization import min_z_lbfgs, min_z_lbfgs_MH_normalized
from scipy.optimize import fmin_l_bfgs_b
from model import model

class ParamDenoising:
    def __init__(self, prior, nb_iter, beta, delta):
        self.prior = prior
        self.nb_iter = nb_iter
        self.beta = beta
        self.delta = delta

    def get_paramDenoising(self):
        return self

class ParamDenoising_dTV(ParamDenoising):
    def __init__(self, prior, nb_iter, beta, delta, theta, eta, eps):
        ParamDenoising.__init__(self, prior, nb_iter, beta, delta)
        self.theta = theta
        self.eta = eta
        self.eps = eps

class ParamDenoising_JTV(ParamDenoising):
    def __init__(self, prior, nb_iter, beta, delta, theta):
        ParamDenoising.__init__(self, prior, nb_iter, beta, delta)
        self.theta = theta



def fullcost(xvect, x_noisy, w, fz, s, beta):
    N = np.shape(x_noisy)[0]
    x = xvect.reshape((N, N), order='F')

    fd = 0.5 * w * np.square(x-x_noisy)
    fd = np.sum(fd)
    gd = w * (x-x_noisy)
    gd = gd.flatten(order='F')

    fr = 0.5 * np.square(s*x-fz)
    fr = np.sum(fr)
    gr = s * (s*x-fz)
    gr = gr.flatten(order='F')

    f = fd + beta * fr
    g = gd + beta * gr

    return f, g


def denoising_Uconnect(x_hat_all, paramImg, paramReg, paramDenoising):
    s_hat_all = np.load("/home/zhihan/recon_results/real_data/s_hat_all.npy").tolist()
    z = 0.01 * np.ones(paramImg.tensor_shape)
    x_all = []
    for i_e in range(len(paramImg.energy_tgt)):
        x_all.append(x_hat_all[i_e].reshape(paramImg.tensor_shape))
    for i in range(50):
        z = min_z_lbfgs(z, x_all, s_hat_all, paramImg, paramReg)
        np.save('/home/zhihan/recon_results/real_data/Uconnect/allunet/z/alpha' + str(paramReg.alpha) + '_beta' + str(
            paramDenoising.beta) + '_' + str(i) + '.npy', z)

        for i_e in range(len(paramImg.energy_tgt)):
            fz = model(z, i_e, paramImg).numpy().reshape(paramImg.img_shape)
            s = s_hat_all[i_e]
            x = (x_hat_all[i_e] + paramDenoising.beta * s * fz) / (1 + paramDenoising.beta * np.square(s))

            x_all[i_e] = x.reshape(paramImg.tensor_shape)

    return x_all

def denoising_MHUconnect(x_hat_all, paramImg, paramReg, paramDenoising):
    # z = 0.01 * np.ones(paramImg.tensor_shape)
    z = np.load('/home/zhihan/data/recon_results/real_data/MHUconnect/initz_alpha'+str(paramReg.alpha)+'.npy')
    x_all = []
    for i_e in range(len(paramImg.energy_tgt)):
        x_all.append(x_hat_all[i_e].reshape(paramImg.tensor_shape))
    for i in range(50):
        z = min_z_lbfgs_MH_normalized(z, x_all, paramImg, paramReg)
        np.save('/home/zhihan/data/recon_results/real_data/MHUconnect/z/alpha' + str(paramReg.alpha) + '_beta' + str(
            paramDenoising.beta) + '_' + str(i) + '.npy', z)
        output = paramImg.model(z)
        for i_e in range(len(paramImg.energy_tgt)):
            #w = omega_all[i_e]
            fz = output[i_e].numpy().reshape(paramImg.img_shape)
            mu_max = paramImg.mu_max[i_e]
            x = (x_hat_all[i_e] + paramDenoising.beta * mu_max * fz) / (1 + paramDenoising.beta)

            x_all[i_e] = x.reshape(paramImg.tensor_shape)
        np.save('/home/zhihan/data/recon_results/real_data/MHUconnect/x/' + str(paramReg.alpha) + '_beta' + str(
            paramDenoising.beta) + '_' + str(i) + '.npy', x_all)

    return x_all

def diff_operator2D(x):
    Dx = np.zeros([np.shape(x)[0],np.shape(x)[1],8])
    l = np.array([0, 1, -1])
    vect = np.zeros((9, 2))
    ii = 0
    for i in range(3):
        for j in range(3):
            vect[ii, :] = np.array([l[i], l[j]])
            ii = ii + 1
    vect = np.delete(vect, 0, axis=0)
    for i in range(np.size(vect, 0)) :
        shift_vect = vect[i,:].astype(int)
        im_shifted = np.roll(np.roll(x, shift_vect[0], axis=0), shift_vect[1], axis=1)
        Dx[:,:,i] = (x - im_shifted) / np.linalg.norm(shift_vect)

    return Dx

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





def denoise_dTV(x_noisy, x_prior,paramDenoising_dTV):
    N = np.shape(x_noisy)[0]
    kappa = paramDenoising_dTV.beta
    #kappa = 0.0005
    theta = paramDenoising_dTV.theta
    #theta = 1
    #eps = 0.00001
    eps = paramDenoising_dTV.eps
    # compute directional information xi from x_prior
    grad_x_prior = compute_grad(x_prior)
    norm_grad_x_prior_eps = np.sqrt(grad_x_prior[:,:,0]**2 + grad_x_prior[:,:,1]**2 + eps**2)
    eta = paramDenoising_dTV.eta
    #eta = 0.7
    xi = np.zeros(np.shape(grad_x_prior))
    xi[:, :, 0] = eta * grad_x_prior[:, :, 0] / norm_grad_x_prior_eps
    xi[:, :, 1] = eta * grad_x_prior[:, :, 1] / norm_grad_x_prior_eps

    # Computing L cf Appendix A of Sidky et al., 2012rf
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

    z = np.zeros(np.shape(computeP(compute_grad(xbar), xi)))

    for i in range(paramDenoising_dTV.nb_iter):
        z_temp = z + sigma * computeP(compute_grad(xbar), xi)
        normZ = np.sqrt(z_temp[:, :, 0] ** 2 + z_temp[:, :, 1] ** 2)
        z = np.zeros(np.shape(z_temp))
        z[:, :, 0] = np.where(normZ > kappa, kappa, normZ) * (z_temp[:, :, 0] / normZ)
        z[:, :, 1] = np.where(normZ > kappa, kappa, normZ) * (z_temp[:, :, 1] / normZ)
        z = np.where(np.isnan(z), 0, z)

        x_old = copy.deepcopy(x)
        x_temp = x_old - tau * compute_grad_back(computeP(z, xi))
        x = (x_noisy + x_temp / tau) / (1 + 1 / tau)
        xbar = x + theta * (x - x_old)

    return x


def denoise_JTV(x_noisy_all, paramDenoising_JTV):
    nb_energy = len(x_noisy_all)
    N = np.shape(x_noisy_all[0])[0]
    kappa = paramDenoising_JTV.beta
    theta = paramDenoising_JTV.theta

    x = np.random.rand(N, N)

    prior = paramDenoising_JTV.prior

    for k in range(100):
        if (prior == 'JTV1'):
            x = diff_back2D(diff_operator2D(x))
            normalisation = np.sqrt(np.sum(x ** 2))
            x = x / normalisation
            s = np.sqrt(np.sum(diff_operator2D(x) ** 2))
        elif (prior == 'JTV2'):
            x = compute_grad_back(compute_grad(x))
            normalisation = np.sqrt(np.sum(x ** 2))
            x = x / normalisation
            s = np.sqrt(np.sum(compute_grad(x) ** 2))
    L = s

    sigma = 0.8 * 1 / L
    tau = 0.8 * 1 / L

    x = [np.zeros([N, N]) for i_e in range(nb_energy)]
    xbar = copy.deepcopy(x)

    if (prior == 'JTV1'):
        z = [np.zeros(np.shape(diff_operator2D(xbar[i_e]))) for i_e in range(nb_energy)]
    elif (prior == 'JTV2'):
        z = [np.zeros(np.shape(compute_grad(xbar[i_e]))) for i_e in range(nb_energy)]

    for i in range(paramDenoising_JTV.nb_iter):
        print("Iteration %(i)d" % {"i": i})
        z_temp = []
        nor = 0
        if (prior == 'JTV1'):
            for i_e in range(nb_energy):
                temp = z[i_e] + sigma * diff_operator2D(xbar[i_e])
                z_temp.append(temp)
                nor = nor + temp ** 2
            normZ = np.sqrt(nor)
        elif (prior == 'JTV2'):
            for i_e in range(nb_energy):
                temp = z[i_e] + sigma * compute_grad(xbar[i_e])
                z_temp.append(temp)
                nor = nor + temp[:, :, 0] ** 2 + temp[:, :, 1] ** 2
                #nor = nor + temp ** 2
            normZ = np.zeros(np.shape(temp))
            normZ[:, :, 0] = np.sqrt(nor)
            normZ[:, :, 1] = np.sqrt(nor)
            #normZ = np.sqrt(nor)
        for i_e in range(nb_energy):
            temp = np.where(normZ>kappa,kappa,normZ) * (z_temp[i_e] / normZ)
            z[i_e] = np.where(np.isnan(temp),0,temp)
        x_old = copy.deepcopy(x)
        x_temp = [[] for i_e in range(nb_energy)]
        if (prior == 'JTV1'):
            for i_e in range(nb_energy):
                x_temp[i_e] = x_old[i_e] - tau * diff_back2D(z[i_e])
        elif (prior == 'JTV2'):
            for i_e in range(nb_energy):
                x_temp[i_e] = x_old[i_e] - tau * compute_grad_back(z[i_e])
        for i_e in range(nb_energy):
            x[i_e] = (x_noisy_all[i_e] + x_temp[i_e]/tau) / (1+1/tau)
            xbar[i_e] = x[i_e] + theta*(x[i_e]-x_old[i_e])

    return x


def fullcost(xvect, x_noisy, paramDenoisisng):
    N = np.shape(x_noisy)[0]
    x = xvect.reshape((N, N), order='F')

    fd = 0.5*np.square(x-x_noisy)
    fd = np.sum(fd)
    gd = x-x_noisy
    gd = gd.flatten(order='F')

    fr, gr = imagePrior(x, paramDenoisisng)

    f = fd + paramDenoisisng.beta * fr
    g = gd + paramDenoisisng.beta * gr

    return f, g

def denoise_lbfgs(x_noisy, paramDenoisisng):
    N = np.shape(x_noisy)[0]

    x_init = 0.01 * np.ones([N, N])

    xvect_init = x_init.flatten(order='F')

    bnds = [(0, 0.3)] * xvect_init.size

    fullcost_1var = lambda xvect: fullcost(xvect, x_noisy, paramDenoisisng)
    xvect, y_opt, opt_info = fmin_l_bfgs_b(fullcost_1var, xvect_init, fprime=None, bounds=bnds, m=9, factr=1e7, iprint=99, maxiter=paramDenoisisng.nb_iter)

    x = xvect.reshape((N, N), order='F')
    return x


