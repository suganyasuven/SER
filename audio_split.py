import librosa
import numpy as np

sr = 16000
x, sample_rate = librosa.load('IEMOCAP/session1/anger/3_Ses01F_impro01_F012.wav')
# feature = librosa.effects.split(y=x, top_db=10, frame_length=100, hop_length=160)
# print(feature)
# fea = np.concatenate(feature)
# print(fea)
yt, index = librosa.effects.trim(y=x, top_db=10)
print(librosa.get_duration(x), librosa.get_duration(yt))