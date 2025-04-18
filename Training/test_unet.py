# -- coding: utf-8 --**

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os


print(os.getcwd())

# load model
saved_model_dir = 'saved_models_unet/512/40140'
saved_model_name = 'unet_e700_t5_v5_lr0.0002_512x512_ct_ssim20221023-133323'# 140 keV
saved_model_path = f'{saved_model_dir}/{saved_model_name}'
model = keras.models.load_model(saved_model_path, compile=False)



def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=0.05)


img_shape = [512, 512]


energy_tgt = 140
energy_ref = 40

# use another dataset for test
path_tgt = 'dataset/test/atten_' + str(energy_tgt) + 'KeV.npy'
path_ref = 'dataset/test/atten_' + str(energy_ref) + 'KeV.npy'
img_tgt = np.load(path_tgt)
img_ref = np.load(path_ref)


imgs_tgt = []
imgs_ref = []
s_all = []
for i_slide in range(100):
    x_tgt = img_tgt[i_slide,:,:]
    x_ref = img_ref[i_slide,:,:]
    s = np.sum(x_ref) / np.sum(x_tgt)
    s_all.append(s)
    imgs_tgt.append(x_tgt*s)
    imgs_ref.append(x_ref)
imgs_tgt = np.expand_dims(imgs_tgt, axis=3)
imgs_ref = np.expand_dims(imgs_ref, axis=3)


x_test = imgs_ref
y_test = imgs_tgt


ct_input = x_test[70:71]
ct_tgt = y_test[70:71]
s = s_all[70]


# get prediction image
ct_output = model.predict(ct_input)

# compute loss
loss_fn = tf.keras.losses.MeanSquaredError(reduction="auto")
loss_value1 = loss_fn(ct_output, ct_tgt)

# compute ssim
ssim1 = SSIM(ct_tgt.astype('double'), ct_output.astype('double'))

# display results
print(f'ct loss: {loss_value1}')
print(f'ct ssim: {ssim1[0]}')

ct_output = ct_output/s
ct_tgt = ct_tgt/s

print(f'loss: {loss_value1}')
print(f'ssim: {ssim1}')

# display the input, the output of Unet and the ground truth
fig = plt.figure(figsize=(80, 30))
rows = 1
cols = 3

fig.add_subplot(rows, cols, 1)
plt.imshow(ct_input.reshape(img_shape), cmap='gray')
plt.clim(0,0.2)
plt.axis('off')
plt.title('CT ref-low energy')

ax2 = fig.add_subplot(rows, cols, 2)
ax2.text(0, 0, f'SSIM: {ssim1[0]:.4f}', color='green', fontsize=10, transform=ax2.transAxes,
         verticalalignment='bottom')
plt.imshow(ct_output.reshape(img_shape), cmap='gray')
plt.clim(0,0.2)
plt.axis('off')
plt.title('Obtained CT high energy')

fig.add_subplot(rows, cols, 3)
plt.imshow(ct_tgt.reshape(img_shape), cmap='gray')
plt.clim(0,0.2)
plt.axis('off')
plt.title('CT ground-truth high energy')

plt.show()
