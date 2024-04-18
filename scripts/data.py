import numpy as np 

data = np.load('D:\\Semester 8\\AudioGeneration\\stft_spectrograms.npz')
loaded_spectrograms = [data[f'{i}'] for i in range(len(data))]

# shape of a STFT spectrogram = (1025, 2584)
# 1515 spectograms
print(len(loaded_spectrograms))