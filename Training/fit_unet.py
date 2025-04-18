import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from unet import build_unet


def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=0.05)

# target energy (keV)
energy_tgt = 140 # 60 or 80 or 100 or 120 or 140
# reference energy: lowest energy (keV)
energy_ref = 40

# load datasets for training from img_ref to img_tgt
path_tgt = 'dataset/train/atten_' + str(energy_tgt) + 'KeV.npy'
path_ref = 'dataset/train/atten_' + str(energy_ref) + 'KeV.npy'
img_tgt = np.load(path_tgt)
img_ref = np.load(path_ref)

# organize the datasets and add a scale to each slide of target image: because the attenuation value for high energies is much smaller than the reference
imgs_tgt = []
imgs_ref = []
for i_slide in range(np.shape(img_tgt)[0]):
    x_tgt = img_tgt[i_slide,:,:]
    x_ref = img_ref[i_slide,:,:]
    s = np.sum(x_ref) / np.sum(x_tgt)
    imgs_tgt.append(x_tgt*s)
    imgs_ref.append(x_ref)
imgs_tgt = np.expand_dims(imgs_tgt, axis=3)
imgs_ref = np.expand_dims(imgs_ref, axis=3)


x_train = imgs_ref
y_train = imgs_tgt

x_val = imgs_ref
y_val = imgs_tgt


# set hyperparameters
learning_rate = 2e-4
epoch = 700
train_batch_size = 5
val_batch_size = 5

# get model
input_shape = (512, 512, 1)
model = build_unet(input_shape)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='mean_squared_error', metrics=[SSIM])

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") + \
          f"unet_e{epoch}_t{train_batch_size}_v{val_batch_size}_lr{learning_rate}"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(x=x_train, y=y_train, epochs=epoch, batch_size=train_batch_size, validation_data=(x_val, y_val), shuffle=True,
                    callbacks=[tensorboard_callback])

saved_model_dir = 'saved_models_unet'
saved_model_name = f'unet_e{epoch}_t{train_batch_size}_v{val_batch_size}_lr{learning_rate}_128x128_ct_ssim' + \
                   datetime.now().strftime("%Y%m%d-%H%M%S")
saved_model_path = f'{saved_model_dir}/{saved_model_name}'
model.save(saved_model_path)
keras.backend.clear_session()
