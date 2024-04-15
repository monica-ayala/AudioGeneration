import librosa
import numpy as np
import matplotlib.pyplot as plt
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

output_directory = 'spectrogram_images'
directory = 'D:\\Semester 8\\AudioGeneration\\dataset'
all_spectrograms = process_all_files(directory, output_directory)
