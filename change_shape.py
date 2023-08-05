import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom

x_train = np.load('n_x_train_speech.npy')
y_train = np.load('n_y_train_speech.npy')
print(x_train.shape)
print(y_train.shape)

zoom_factors = [1, 60 / 60, 60 / 126, 1]

rescaled_mel_spectrograms = np.empty((11976, 60, 60, 1))

for i in range(x_train.shape[0]):

    rescaled_mel_spectrograms[i, :, :, 0] = zoom(x_train[i, :, :, 0], zoom_factors[1:3])

#x_train = [:, :, ::5, :]
out_file = 'x_train_60.npy'

print(rescaled_mel_spectrograms.shape)

np.save('x_train_60.npy', rescaled_mel_spectrograms)
#x_train = tf.image.resize(x_train, [1500, 28, 28, 1])
#print(x_train.shape)
