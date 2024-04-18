import librosa
import soundfile as sf
import numpy as np

def normalize_spectrogram(spectrogram):
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    normalized = (spectrogram - min_val) / (max_val - min_val)
    return normalized

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
    magnitude, phase = librosa.magphase(stft)
    stft_reconstructed = magnitude * phase
    y_reconstructed = librosa.istft(stft_reconstructed)
    sf.write(output_path, y_reconstructed, sr)

audio_to_stft_to_audio('D:\\Semester 8\\AudioGeneration\\dataset\\188.mp3', 'new.wav')

