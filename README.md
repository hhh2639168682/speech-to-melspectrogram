# speechcommand-to-melspectrogram
This file is uesd for translate wav file to melspectrgram which is used in input to image processing neural network(CNN,RNN....)
# Download dataset(Speechcommand and ESC-50)
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
tar -xf speech_commands_v0.01.tar.gz
wget https://github.com/karoldvl/ESC-50/archive/master.zip
unzip master.zip
//also support speech_commandV2
# Environment set
pip install requirements.txt
# SpeechCommand's PATH
SAVE_PATH = "data_quantum/" 

train_audio_path = '../dataset/'
# ESC-50 path
train_audio_path = 'ESC-50-master/audio'

# Modify the shape
use the change_shape.py

# Also Supporting modify some numbers in geting melspectrogram

# Reference
https://github.com/huckiyang/QuantumSpeech-QCNN
