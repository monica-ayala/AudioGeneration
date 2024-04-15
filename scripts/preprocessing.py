import librosa
import librosa.effects
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

def load_and_save_melspectrogram(file_path, output_dir, file_name, n_mels=128, n_fft=2048, hop_length=512, sr=None):
    
    """
    Load an audio file, compute its Mel-spectrogram, and save the spectrogram as an image.

    Parameters:
    - file_path: str, path to the audio file.
    - output_dir: str, directory where spectrogram images will be saved.
    - file_name: str, name for the output image file.
    - n_mels: int, number of Mel bands to generate.
    - n_fft: int, length of the FFT window.
    - hop_length: int, number of samples between successive frames.
    - sr: int or None, sample rate of the audio file; if None, the file's original sample rate is used.
    """

    audio, sample_rate = librosa.load(file_path, sr=sr)
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"{file_name}.png"))
    plt.close()

    return S_dB

def process_all_files(directory, output_directory):
    spectrograms = []
    for filename in os.listdir(directory):
        if filename.endswith('.mp3'):
            file_path = os.path.join(directory, filename)
            file_name = os.path.splitext(filename)[0]  
            spectrogram = load_and_save_melspectrogram(file_path, output_directory, file_name)
            spectrograms.append(spectrogram)
    return spectrograms

## Normalization

def normalize_spectrogram(spectrogram):
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    return (spectrogram - min_val) / (max_val - min_val)

## Data Augmentation

def pitch_shift_spectrogram(spectrogram, sample_rate, n_steps):
    audio = librosa.feature.inverse.mel_to_audio(spectrogram, sr=sample_rate, n_fft=2048, hop_length=512)
    audio_shifted = librosa.effects.pitch_shift(audio, sample_rate, n_steps)
    spectrogram_shifted = librosa.feature.melspectrogram(y=audio_shifted, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128)
    return librosa.power_to_db(spectrogram_shifted, ref=np.max)

def add_noise_spectrogram(spectrogram, noise_level=0.005):
    noise = np.random.randn(*spectrogram.shape) * noise_level
    return spectrogram + noise

## Saving Data

def save_spectrograms_to_hdf5(spectrograms, file_path):
    with h5py.File(file_path, 'w') as f:
        for i, spectrogram in enumerate(spectrograms):
            f.create_dataset(f'spectrogram_{i}', data=spectrogram, compression="gzip", compression_opts=9)

output_directory = 'spectrogram_images_test'
directory = 'D:\\Semester 8\\AudioGeneration\\test'
all_spectrograms = process_all_files(directory, output_directory)
spectrograms_normalized = [normalize_spectrogram(s) for s in all_spectrograms]
save_spectrograms_to_hdf5(spectrograms_normalized, 'spectrograms.hdf5')