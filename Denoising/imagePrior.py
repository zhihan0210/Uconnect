import numpy as np
import time

def quadr(x):
    psi0 = 0.5*np.square(x)
    psi1 = x
    return psi0, psi1

def huber(x, delta):
    psi0 = np.where(np.abs(x)<=delta,0.5*np.square(x),delta*np.abs(x)-0.5*np.square(delta))
    psi1 = np.where(np.abs(x)<=delta,x,np.sign(x)*delta)
    return psi0, psi1

def TV(x, delta):
    psi0 = np.sqrt(np.square(x)+delta)
    psi1 = np.true_divide(x,np.sqrt(np.square(x)+delta))
    return psi0, psi1



def imagePrior(im,param):
    
    l = np.array([0,1,-1])
    
    vect = np.zeros((9,2))
    
    ii = 0
    for i in range(3) :
        for j in range(3) :
            vect[ii,:] = np.array([l[i],l[j]])
            ii = ii+1
    vect = np.delete(vect,0,axis=0)
    
    F = 0
    D = 0
    
    for i in range(np.size(vect, 0)) :
        shift_vect = vect[i,:].astype(int)
        im_shifted = np.roll(np.roll(im, shift_vect[0], axis=0), shift_vect[1], axis=1)
        
        omega = np.ones(np.shape(im))/np.linalg.norm(shift_vect)
        
        def switchPrior(prior):
            return {
                #'TV': lambda: TV(im-im_shifted,param.delta),
                'Huber': lambda: huber(im-im_shifted,param.delta),
                'quadratic': lambda: quadr(im-im_shifted),
            }.get(prior, lambda: None)()
        psi0, psi1 = switchPrior(param.prior)
        
        F = F + omega*psi0
        D = D + 2*omega*psi1
              
    f = np.sum(F)
    g = D.flatten(order='F')
    
    return f, g


# def latent_prior(z, param):
#     # Define penalty terms for the latent space
#     def switchPrior(prior):
#         return {
#             'L2': (np.sum(np.square(z)), 2 * z),
#             'L1': (np.sum(np.abs(z)), np.sign(z)),
#             # Add other penalty terms as needed
#         }.get(prior, lambda: None)()
#
#     penalty, gradient = switchPrior(param.prior)
#
#     # Return the penalty term and its gradient
#     return penalty, gradient

def imagePrior_SPS(im_old,param):

    l = np.array([0, 1, -1])

    vect = np.zeros((9, 2))

    ii = 0
    for i in range(3):
        for j in range(3):
            vect[ii, :] = np.array([l[i], l[j]])
            ii = ii + 1
    vect = np.delete(vect, 0, axis=0)

    omega_reg = 0
    im_reg = 0

    for i in range(np.size(vect, 0)):
        shift_vect = vect[i, :].astype(int)
        im_shifted = np.roll(np.roll(im_old, shift_vect[0], axis=0), shift_vect[1], axis=1)
        diff_old = im_old - im_shifted

        def switchPrior(prior):
            return {
                #'TV': lambda: TV(diff_old, param.delta),
                'Huber': lambda: huber(diff_old, param.delta),
                'quadratic': lambda: quadr(diff_old),
            }.get(prior, lambda: None)()

        psi0, psi1 = switchPrior(param.prior)

        omega = np.ones(np.shape(im_old)) / np.linalg.norm(shift_vect)
        ratio = psi1/diff_old
        ratio = np.where(diff_old==0,1,ratio)



        omega_surro = omega*ratio
        #omega_surro = np.where(diff_old==0,omega,omega_surro)

        omega_reg = omega_reg + omega_surro
        im_reg = im_reg + (im_old + im_shifted) * omega_surro

    im_reg = 0.5*im_reg/omega_reg

    return im_reg, omega_reg
