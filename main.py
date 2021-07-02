import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read as read_wav

path = 'song_1001.wav'
sampling_rate, _ = read_wav(path)

x = librosa.load(path, sampling_rate)[0]
S = librosa.feature.melspectrogram(x, sr=sampling_rate, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)
mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=12)

delta2_mfcc = librosa.feature.delta(mfcc, order=2)

plt.figure(figsize=(96, 4))
librosa.display.specshow(delta2_mfcc, sr=sampling_rate, x_axis='time')
plt.ylabel('MFCC coeffs')
plt.xlabel('Time')
plt.title('MFCC')
plt.colorbar()
plt.tight_layout()
plt.show()