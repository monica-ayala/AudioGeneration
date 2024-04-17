# import librosa
# import soundfile as sf

# def audio_to_stft_to_audio(audio_path, output_path):
#     # Load audio file
#     y, sr = librosa.load(audio_path, sr=None)  # sr=None to preserve the original sampling rate

#     # Compute the STFT of the audio
#     stft = librosa.stft(y)
#     # Convert the complex-valued STFT to magnitude and phase
#     magnitude, phase = librosa.magphase(stft)

#     # Reconstruct the STFT matrix from magnitude and phase
#     stft_reconstructed = magnitude * phase

#     # Perform the inverse STFT to get the time domain signal
#     y_reconstructed = librosa.istft(stft_reconstructed)

#     # Save the reconstructed audio
#     sf.write(output_path, y_reconstructed, sr)

# # Example usage
# audio_to_stft_to_audio('D:\\Semester 8\\AudioGeneration\\test\\188.mp3', 'new.wav')
# Load the compressed NPZ file

import numpy as np 

data = np.load('D:\\Semester 8\\AudioGeneration\\normalized_stft_spectrograms.npz')
loaded_spectrograms = [data[f'{i}'] for i in range(len(data))]
print(loaded_spectrograms)
