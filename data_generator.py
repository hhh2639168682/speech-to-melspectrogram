import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import warnings
import tensorflow as tf
import librosa.display
import matplotlib.pyplot as plt
from pennylane import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
## Local Definition 
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time as ti
from scipy.ndimage import zoom

labels = [
    'left', 'go', 'yes', 'down', 'up', 'on', 'right', 'no', 'off', 'stop',
]

sr = 16000
port = 100

SAVE_PATH = "data_quantum/" # Data saving folder
train_audio_path = '../dataset/'
sr=16000


def gen_mel(labels, train_audio_path, sr, port):
    all_wave = []
    all_label = []
    for label in tqdm(labels):
        waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
        for num, wav in enumerate(waves, 0):
            y, _ = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = sr)
            if num % port ==0:   # take 1/port samples
                if(len(y)== sr) :
                    mel_feat = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=128, power=1.0, n_mels=60, fmin=40.0, fmax=sr/2)
                    all_wave.append(np.expand_dims(mel_feat, axis=2))
                    all_label.append(label)
    
    return all_wave, all_label

def gen_train(labels, train_audio_path, sr, port):
    all_wave, all_label = gen_mel(labels, train_audio_path, sr, port)

    label_enconder = LabelEncoder()
    y = label_enconder.fit_transform(all_label)
    classes = list(label_enconder.classes_)
    y = keras.utils.to_categorical(y, num_classes=len(labels))

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)
    h_feat, w_feat, _ = x_train[0].shape
    np.save(SAVE_PATH + "n_x_train_speech.npy", x_train)
    np.save(SAVE_PATH + "n_x_test_speech.npy", x_test)
    np.save(SAVE_PATH + "n_y_train_speech.npy", y_train)
    np.save(SAVE_PATH + "n_y_test_speech.npy", y_test)
    print("===== Shape", h_feat, w_feat)

    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
        labels = [
              'left', 'go', 'yes', 'down', 'up', 'on', 'right', 'no', 'off', 'stop',
              ]
        train_audio_path = '../dataset/'
        sr = 16000
        port = 1
        SAVE_PATH = "data_quantum/" # Data saving folder
           
        x_train, x_test, y_train, y_test = gen_train(labels, train_audio_path, sr, port)
        print("x_train shape:", x_train.shape)

