# -- coding: utf-8 --**
import numpy as np
from tensorflow import keras

im_shape = [512, 512]
ten_shape = [1, 512, 512, 1]

e_tgt = [40,60,80,100,120,140]
e_ref = 40

# load Unet network for each energy as an attribute "model" of the class ParamImg
saved_model40_path = 'saved_models_unet/512/4040/unet_e700_t5_v5_lr0.0002_512x512_ct_ssim20221024-073504' # 40KeV
saved_model60_path = 'saved_models_unet/512/4060/unet_e700_t5_v5_lr0.0002_512x512_ct_ssim20221022-053236' # 60KeV
saved_model80_path = 'saved_models_unet/512/4080/unet_e700_t5_v5_lr0.0002_512x512_ct_ssim20221022-132928' # 80KeV
saved_model100_path = 'saved_models_unet/512/40100/unet_e700_t5_v5_lr0.0002_512x512_ct_ssim20221022-213531' # 100KeV
saved_model120_path = 'saved_models_unet/512/40120/unet_e700_t5_v5_lr0.0002_512x512_ct_ssim20221023-053405' # 120KeV
saved_model140_path = 'saved_models_unet/512/40140/unet_e700_t5_v5_lr0.0002_512x512_ct_ssim20221023-133323' # 140KeV

list_model = []
list_model.append(keras.models.load_model(saved_model40_path, compile=False))
list_model.append(keras.models.load_model(saved_model60_path, compile=False))
list_model.append(keras.models.load_model(saved_model80_path, compile=False))
list_model.append(keras.models.load_model(saved_model100_path, compile=False))
list_model.append(keras.models.load_model(saved_model120_path, compile=False))
list_model.append(keras.models.load_model(saved_model140_path, compile=False))



class ParamImg:
    def __init__(self, energy_ref=e_ref, energy_tgt=e_tgt, img_shape=im_shape, tensor_shape=ten_shape, model=list_model):
        self.energy_ref = energy_ref
        self.energy_tgt = energy_tgt
        self.img_shape = img_shape
        self.tensor_shape = tensor_shape
        self.model = model

    def get_paramImg(self):
        return self

