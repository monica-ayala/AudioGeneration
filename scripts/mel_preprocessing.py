import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
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

def normalize_spectrogram(spectrogram):
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    return (spectrogram - min_val) / (max_val - min_val)

output_directory = 'spectrogram_imagess'
directory = 'D:\\Semester 8\\AudioGeneration\\test'
all_spectrograms = process_all_files(directory, output_directory)
spectrograms_normalized = [normalize_spectrogram(s) for s in all_spectrograms]