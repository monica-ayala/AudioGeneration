import librosa
import soundfile as sf
import numpy as np

FIXED_MIN_VAL = -80  
FIXED_MAX_VAL = 0 

def normalize_spectrogram(spectrogram):
    normalized = (spectrogram - FIXED_MIN_VAL) / (FIXED_MAX_VAL - FIXED_MIN_VAL)
    return normalized

def inverse_normalize_spectrogram(normalized_spectrogram):
    return normalized_spectrogram * (FIXED_MAX_VAL - FIXED_MIN_VAL) + FIXED_MIN_VAL

def reshape(spectogram):
    if spectogram.shape >= (1024, 2048):
        reshaped = spectogram[:1024, :2048, np.newaxis] 
    else:
        reshaped = np.zeros((1024, 2048, 1))
        reshaped[:spectogram.shape[0], :spectogram.shape[1], 0] = spectogram

    return reshaped

def audio_to_stft_to_audio(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=None)
    stft = librosa.stft(y)
    stft = normalize_spectrogram(stft)
    stft = reshape(stft)
    stft = stft[..., 0]
    stft = inverse_normalize_spectrogram(stft)
    magnitude, phase = librosa.magphase(stft)
    stft_reconstructed = magnitude * phase
    y_reconstructed = librosa.istft(stft_reconstructed)
    sf.write(output_path, y_reconstructed, sr)

audio_to_stft_to_audio('C:\\Users\\mayal\\AudioGeneration\\dataset\\188.mp3', 'new.wav')

