import librosa
import soundfile as sf

def audio_to_stft_to_audio(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=None)
    stft = librosa.stft(y)
    magnitude, phase = librosa.magphase(stft)
    stft_reconstructed = magnitude * phase
    y_reconstructed = librosa.istft(stft_reconstructed)
    sf.write(output_path, y_reconstructed, sr)

audio_to_stft_to_audio('D:\\Semester 8\\AudioGeneration\\test\\188.mp3', 'new.wav')

