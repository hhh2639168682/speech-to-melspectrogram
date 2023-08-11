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






SAVE_PATH = "/data_esc50/" # Data saving folder
label_mapping = {
    0: 'dog', 14: 'chirping_birds', 36: 'vacuum_cleaner', 19: 'thunderstorm', 30: 'door_wood_knock',34: 'can_opening', 9: 'crow', 22: 'clapping', 48: 'fireworks', 41: 'chainsaw', 47: 'airplane', 31: 'mouse_click', 17: 'pouring_water', 45: 'train', 8: 'sheep', 15: 'water_drops', 46: 'church_bells', 37: 'clock_alarm', 32: 'keyboard_typing', 16: 'wind', 25: 'footsteps', 4: 'frog', 3: 'cow', 27: 'brushing_teeth', 43: 'car_horn', 12: 'crackling_fire', 40: 'helicopter', 29: 'drinking_sipping', 10: 'rain', 7: 'insects', 26: 'laughing', 6: 'hen', 44: 'engine', 23: 'breathing', 20: 'crying_baby', 49: 'hand_saw', 24: 'coughing', 39: 'glass_breaking', 28: 'snoring', 18: 'toilet_flush', 2: 'pig', 35: 'washing_machine', 38: 'clock_tick', 21: 'sneezing', 1: 'rooster', 11: 'sea_waves', 42: 'siren', 5: 'cat', 33: 'door_wood_creaks', 13: 'crickets'
}
train_audio_path = 'ESC-50-master/audio'

def gen_mel(label_mapping, train_audio_path, sr, port):
    all_wave = []
    all_label = []
    
    waves = [f for f in os.listdir(train_audio_path) if f.endswith('.wav')]
    print(f"Found {len(waves)} WAV files") 
    for num, wav in enumerate(waves, 0):
        
        category_num = int(wav.split('-')[-1].split('.')[0])
        
        
        label = label_mapping[category_num]
        print(f"Processing file {wav}, category number {category_num}, label {label}")
        y, _ = librosa.load(train_audio_path + '/' + wav, sr=sr)
        print(f"File {wav}, length of y: {len(y)}, sr: {sr}, num % port: {num % port}")
        y_resampled = librosa.resample(y, orig_sr=len(y), target_sr=sr) 

        mel_feat = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=220, power=1.0, n_mels=60, fmin=40.0, fmax=sr/2)
        all_wave.append(np.expand_dims(mel_feat, axis=2))
        all_label.append(label)
    
    return all_wave, all_label

def gen_train(label_mapping, train_audio_path, sr, port):
    all_wave, all_label = gen_mel(label_mapping, train_audio_path, sr, port)

    label_enconder = LabelEncoder()
    y = label_enconder.fit_transform(all_label)
    classes = list(label_enconder.classes_)
    y = keras.utils.to_categorical(y, num_classes=len(label_mapping))
    print(f"Total samples: {len(all_wave)}")
    print(f"Shape of y: {y.shape}")

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)
    h_feat, w_feat, _ = x_train[0].shape
    np.save( "n_x_train_speech.npy", x_train)
    np.save( "n_x_test_speech.npy", x_test)
    np.save( "n_y_train_speech.npy", y_train)
    np.save( "n_y_test_speech.npy", y_test)
    print("===== Shape", h_feat, w_feat)

    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
        label_mapping = {
        0: 'dog', 14: 'chirping_birds', 36: 'vacuum_cleaner', 19: 'thunderstorm', 30: 'door_wood_knock',34: 'can_opening', 9: 'crow', 22: 'clapping', 48: 'fireworks', 41: 'chainsaw', 47: 'airplane', 31: 'mouse_click', 17: 'pouring_water', 45: 'train', 8: 'sheep', 15: 'water_drops', 
          46: 'church_bells', 37: 'clock_alarm', 32: 'keyboard_typing', 16: 'wind', 25: 'footsteps', 4: 'frog', 3: 'cow', 27: 'brushing_teeth', 43: 'car_horn', 12: 'crackling_fire', 40: 'helicopter', 29: 'drinking_sipping', 10: 'rain', 7: 'insects', 26: 'laughing', 6: 'hen', 44: 'engine', 
          23: 'breathing', 20: 'crying_baby', 49: 'hand_saw', 24: 'coughing', 39: 'glass_breaking', 28: 'snoring', 18: 'toilet_flush', 2: 'pig', 35: 'washing_machine', 38: 'clock_tick', 21: 'sneezing', 1: 'rooster', 11: 'sea_waves', 42: 'siren', 5: 'cat', 33: 'door_wood_creaks', 13: 'crickets'
        }
        train_audio_path = 'ESC-50-master/audio'
        sr = 16000
        port = 1
        SAVE_PATH = "data_esc50/" # Data saving folder
           
        x_train, x_test, y_train, y_test = gen_train(label_mapping, train_audio_path, sr, port)
        print("x_train shape:", x_train.shape)

