# -- coding: utf-8 --**
import numpy as np
from tensorflow import keras

im_shape = [512, 512]
ten_shape = [1, 512, 512, 1]

# pixel_size = 0.075 # unit: cm/pixel
pixel_size = 0.0779296875 # unit: cm/pixel
mu_cm_water = [0.2683,0.2059,0.1837,0.1707,0.1614,0.1538] # unit: /cm
mu_cm_air = [0.0786,0.0609,0.0545,0.0507,0.0479,0.0457] # unit: /cm
mu_cm_max = []
mu_pixel_max = []
for i_e in range(6):
    mu_max = 3071 * (mu_cm_water[i_e] - mu_cm_air[i_e]) / 1000 + mu_cm_water[i_e]
    mu_cm_max.append(mu_max) # unit: /cm
    mu_pixel_max.append(mu_max * pixel_size)



e_tgt = [40,60,80,100,120,140]
e_ref = 40


saved_model_path = '/home2/zwang/denoise/model/unet_e3000_t5_v5_lr0.0002_512x512_ct_ssim20240922-034637'

model = keras.models.load_model(saved_model_path, compile=False)




class ParamImg:
    def __init__(self, energy_ref=e_ref, energy_tgt=e_tgt, img_shape=im_shape, tensor_shape=ten_shape, model=model, mu_max = mu_pixel_max):
        self.energy_ref = energy_ref
        self.energy_tgt = energy_tgt
        self.img_shape = img_shape
        self.tensor_shape = tensor_shape
        self.model = model
        self.mu_max = mu_max

    def get_paramImg(self):
        return self

def normalize_img(im, mu_max):
    nor = np.zeros_like(im)  # Create an array to store normalized values
    nor[im / mu_max > 1] = 1  # Set values greater than 1 to 1
    nor[im / mu_max < 0] = 0  # Set values less than 0 to 0
    nor[(im / mu_max >= 0) & (im / mu_max <= 1)] = im[(im / mu_max >= 0) & (im / mu_max <= 1)] / mu_max  # Set other values to data/mu_max
    return nor
