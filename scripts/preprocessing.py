import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import h5py
import soundfile as sf
import os

def segment_and_save_melspectrogram(file_path, output_dir, file_name, segment_length=30, n_mels=128, n_fft=2048, hop_length=512, sr=None):

    audio, sample_rate = librosa.load(file_path, sr=sr)
    duration = librosa.get_duration(y=audio, sr=sample_rate)
    total_segments = int(np.ceil(duration / segment_length))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    spectrograms = []

    for i in range(total_segments):
        start_sample = i * segment_length * sample_rate
        end_sample = start_sample + segment_length * sample_rate
        segment = audio[start_sample:end_sample] if end_sample < len(audio) else np.pad(audio[start_sample:], (0, end_sample - len(audio)), 'constant')

        S = librosa.feature.melspectrogram(y=segment, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-frequency spectrogram - Segment {i+1}')
        plt.tight_layout()
        
        segment_file_name = f"{file_name}_segment_{i+1}.png"
        plt.savefig(os.path.join(output_dir, segment_file_name))
        plt.close()

        spectrograms.append(S_dB)

    return spectrograms

def process_all_files(directory, output_directory):
    all_spectrograms = []
    for filename in os.listdir(directory):
        if filename.endswith('.mp3'):
            file_path = os.path.join(directory, filename)
            file_name = os.path.splitext(filename)[0]  
            spectrograms = segment_and_save_melspectrogram(file_path, output_directory, file_name)
            all_spectrograms.extend(spectrograms)
    return all_spectrograms

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

output_directory = 'spectrogram_imagess'
directory = 'D:\\Semester 8\\AudioGeneration\\test'
all_spectrograms = process_all_files(directory, output_directory)
# spectrograms_normalized = [normalize_spectrogram(s) for s in all_spectrograms]
# save_spectrograms_to_hdf5(spectrograms_normalized, 'spectrograms.hdf5')

import librosa
import numpy as np

# Assuming `S_dB` is your dB-scaled mel spectrogram
def reconstruct_audio(S_dB, sr=22050, n_iter=32):
    # Convert dB-scaled mel spectrogram back to power mel spectrogram
    S = librosa.db_to_power(S_dB)
    
    # Invert the mel spectrogram to get back to linear frequency spectrogram
    S_inv = librosa.feature.inverse.mel_to_stft(S, sr=sr)
    
    # Use the Griffin-Lim algorithm to estimate the phase
    y_inv = librosa.griffinlim(S_inv, n_iter=n_iter)
    
    return y_inv

# Load the spectrogram (replace this with how you actually load your spectrogram)
S_dB = all_spectrograms[0]
# Reconstruct the audio
reconstructed_audio = reconstruct_audio(S_dB)

# Save or play the reconstructed audio
sf.write('reconstructed_audio.wav', reconstructed_audio, 22050)

