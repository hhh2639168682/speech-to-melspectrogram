# speechcommand-to-melspectrogram
This file is uesd for translate wav file to melspectrgram which is used in input to image processing neural network(CNN,RNN....)
# Download dataset(Speechcommand)
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

tar -xf speech_commands_v0.01.tar.gz
//also support speech_commandV2
# Environment set
pip install requirements.txt
# PATH
SAVE_PATH = "data_quantum/" 

train_audio_path = '../dataset/'

# Modify the shape
use the change_shape.py
