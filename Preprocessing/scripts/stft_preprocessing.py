import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf

FIXED_MIN_VAL = 0  
FIXED_MAX_VAL = 247

def segment_and_save_stft(file_path, output_dir, file_name, segment_length=13, n_fft=1024, hop_length=1024, sr=None):
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
        stft = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length)
        stft_dB = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(stft_dB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'STFT Magnitude - Segment {i+1}')
        plt.tight_layout()
        segment_file_name = f"{file_name}_segment_{i+1}.png"
        plt.savefig(os.path.join(output_dir, segment_file_name))
        plt.close()
        stft_abs = np.abs(stft)
        spectrograms.append(stft_abs)
        
    return spectrograms

def normalize_spectrogram(spectrogram):
    cleaned_spectrogram = np.nan_to_num(spectrogram, nan=0.0)
    normalized_spectrogram = (cleaned_spectrogram - FIXED_MIN_VAL) / (FIXED_MAX_VAL - FIXED_MIN_VAL)
    
    return normalized_spectrogram

def reshape(spectogram):
    if spectogram.shape >= (512, 512):
        reshaped = spectogram[:512, :512, np.newaxis] 
    else:
        reshaped = np.zeros((512, 512, 1))
        reshaped[:spectogram.shape[0], :spectogram.shape[1], 0] = spectogram

    return reshaped

def process_all_files(directory, output_directory):
    all_spectrograms = {}
    i = 0
    
    for filename in os.listdir(directory):
        if filename.endswith('.mp3'):
            file_path = os.path.join(directory, filename)
            file_name = os.path.splitext(filename)[0]
            spectrograms = segment_and_save_stft(file_path, output_directory, file_name)
            for spectrogram in spectrograms:
                normalized_spectrogram = normalize_spectrogram(spectrogram)

                if not np.isnan(normalized_spectrogram).any():
                    reshaped_spectrogram = reshape(normalized_spectrogram)
                    all_spectrograms[f'{i}'] = reshaped_spectrogram  
                    i += 1
                    print(i)
                    
    return all_spectrograms

output_directory = 'stft_images_2'
directory = 'C:\\Users\\mayal\\AudioGeneration\\mini_dataset'
all_spectrograms = process_all_files(directory, output_directory)
np.savez_compressed('data.npz', **all_spectrograms)
