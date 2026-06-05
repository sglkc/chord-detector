import librosa
import matplotlib.pyplot as plt
import numpy as np

fmin = librosa.note_to_hz('C3')
n_bins = 12 * 3
hop_length = 512

y, sr = librosa.load("./datasets/normal/A_major_4/A_major_4-1.wav", sr=None)
cqt = librosa.cqt(y=y, sr=sr, fmin=fmin, n_bins=n_bins, bins_per_octave=12, hop_length=hop_length)
cqt = np.abs(cqt)

librosa.display.specshow(cqt, x_axis='time', y_axis='cqt_note', bins_per_octave=12, fmin=fmin, hop_length=hop_length/2)
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
