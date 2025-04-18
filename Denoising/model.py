import tensorflow as tf

# obtain f(x_ref) with f=Id for 40 keV and f=Unet for other energies
def model(input_img, i_e, paramImg):
    output_img = paramImg.model[i_e](input_img)
    return output_img
