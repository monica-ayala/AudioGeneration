import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf

def segment_and_save_stft(file_path, output_dir, file_name, segment_length=30, n_fft=2048, hop_length=512, sr=None):
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

        # Compute the STFT
        stft = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length)
        # Convert to dB
        stft_dB = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(stft_dB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'STFT Magnitude - Segment {i+1}')
        plt.tight_layout()
        
        segment_file_name = f"{file_name}_segment_{i+1}.png"
        plt.savefig(os.path.join(output_dir, segment_file_name))
        plt.close()

        # Append the dB spectrogram as a tuple with its key
        spectrograms.append(stft_dB)

    return spectrograms

def process_all_files(directory, output_directory):
    all_spectrograms = {}
    i = 0
    for filename in os.listdir(directory):
        if filename.endswith('.mp3'):
            file_path = os.path.join(directory, filename)
            file_name = os.path.splitext(filename)[0]
            spectrograms = segment_and_save_stft(file_path, output_directory, file_name)
            for spectrogram in spectrograms:
                all_spectrograms[f'{i}'] = spectrogram  
                i += 1

    return all_spectrograms

def normalize_spectrogram(spectrogram):
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    return (spectrogram - min_val) / (max_val - min_val)

output_directory = 'stft_images'
directory = 'D:\\Semester 8\\AudioGeneration\\dataset'
all_spectrograms = process_all_files(directory, output_directory)
spectrograms_normalized = {key: normalize_spectrogram(spect) for key, spect in all_spectrograms.items()}
np.savez_compressed('normalized_stft_spectrograms.npz', **spectrograms_normalized)
