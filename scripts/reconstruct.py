import librosa
import soundfile as sf
import numpy as np

FIXED_MIN_VAL = 0  
FIXED_MAX_VAL = 247

def normalize_spectrogram(spectrogram):
    normalized = (spectrogram - FIXED_MIN_VAL) / (FIXED_MAX_VAL - FIXED_MIN_VAL)
    return normalized

def inverse_normalize_spectrogram(normalized_spectrogram):
    return normalized_spectrogram * (FIXED_MAX_VAL - FIXED_MIN_VAL) + FIXED_MIN_VAL

def reshape(spectogram):
    if spectogram.shape >= (512, 512):
        reshaped = spectogram[:512, :512, np.newaxis] 
    else:
        reshaped = np.zeros((512, 512, 1))
        reshaped[:spectogram.shape[0], :spectogram.shape[1], 0] = spectogram

    return reshaped

    # stft = reshape(stft_dB)

def specto_to_audio(stft, output_path):
    stft = stft[:, :, 0]
    stft = inverse_normalize_spectrogram(stft)
    magnitude, phase = librosa.magphase(stft)
    stft_reconstructed = magnitude * phase
    y_reconstructed = librosa.istft(stft_reconstructed, n_fft=1024, hop_length=1024)
    sf.write(output_path, y_reconstructed, 44100)

    
    
def audio_to_stft_to_audio(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=None)
    start_sample = 13 * sr
    end_sample = start_sample + 13 * sr
    segment = y[start_sample:end_sample]
    stft = librosa.stft(segment, n_fft=1024, hop_length=1024)
    stft = normalize_spectrogram(stft)
    stft = reshape(stft)
    print(stft)
    stft = inverse_normalize_spectrogram(stft)
    stft = stft[:, :, 0]
    magnitude, phase = librosa.magphase(stft)
    stft_reconstructed = magnitude * phase
    y_reconstructed = librosa.istft(stft_reconstructed, n_fft=1024, hop_length=1024)
    sf.write(output_path, y_reconstructed, sr)

#audio_to_stft_to_audio('C:\\Users\\mayal\\AudioGeneration\\dataset\\188.mp3', 'new.wav')

data = np.load('dataset.npz')
spectos = np.array([data[f'{i}'] for i in range(len(data))])
print(spectos.shape)
specto_to_audio(spectos[0], 'new.wav')