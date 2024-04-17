import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def segment_and_save_melspectrogram(file_path, file_name, segment_length=30, n_mels=128, n_fft=2048, hop_length=512, sr=None):
    audio, sample_rate = librosa.load(file_path, sr=sr)
    duration = librosa.get_duration(y=audio, sr=sample_rate)
    total_segments = int(np.ceil(duration / segment_length))

    spectrograms = {}
    for i in range(total_segments):
        start_sample = i * segment_length * sample_rate
        end_sample = start_sample + segment_length * sample_rate
        segment = audio[start_sample:end_sample] if end_sample < len(audio) else np.pad(audio[start_sample:], (0, end_sample - len(audio)), 'constant')

        S = librosa.feature.melspectrogram(y=segment, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)
        spectrograms[f"{file_name}_segment_{i+1}"] = S_dB

    return spectrograms

def process_all_files(directory, output_file):
    all_spectrograms = {}
    for filename in os.listdir(directory):
        if filename.endswith('.mp3'):
            file_path = os.path.join(directory, filename)
            file_name = os.path.splitext(filename)[0]
            spectrograms = segment_and_save_melspectrogram(file_path, file_name)
            all_spectrograms.update(spectrograms)
    
    np.savez_compressed(output_file, **all_spectrograms)

def normalize_spectrogram(spectrogram):
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    return (spectrogram - min_val) / (max_val - min_val)

directory = 'D:\\Semester 8\\AudioGeneration\\dataset'
output_file = 'D:\\Semester 8\\AudioGeneration\\spectrograms.npz'
process_all_files(directory, output_file)