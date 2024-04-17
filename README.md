# AudioGeneration
---
**TC3002B: Desarrollo de aplicaciones avanzadas de ciencias computacionales (Gpo 201)**

**Mónica Andrea Ayala Marrero - A01707439**

---
**About this dataset:**

This dataset contains 189 songs from Taylor Swift's discography in mp3 format. The songs were uploaded by different users to the https://archive.org/ platform and compiled manually by me.

**Sources for each album:**
- ***Taylor Swift*** [[link]](https://archive.org/details/cd_taylor-swift_taylor-swift/disc1/01.+Taylor+Swift+-+Tim+McGraw.flac) | 11 songs
- ***Fearless*** [[link]](https://archive.org/details/Fearless-Taylors-Version-Taylor-Swift) | 26 songs
- ***Speak Now*** [[link]](https://archive.org/details/Speak-Now-Taylors-Version-Taylor-Swift) | 22 songs
- ***Red*** [[link]](https://archive.org/details/Red-Album-Taylor-Swift-Taylors-Version) | 28 songs
- ***1989*** [[link]](https://archive.org/details/1989-taylors-version) | 21 songs
- ***Reputation*** [[link]](https://archive.org/details/reputation-cd) | 15 songs
- ***Lover*** [[link]](https://archive.org/details/lover-cd/14+Audio+Track.aiff) | 18 songs
- ***Evermore*** [[link]](https://archive.org/details/happiness_20240409) | 17 songs
- ***Folklore*** [[link]](https://archive.org/details/epiphany_20240407) | 17 songs
- ***Midnights*** [[link]](https://archive.org/details/01.-lavender-haze) | 14 songs

**Information:**

*Format*: MP3 Audio Files

*Date*: Compiled on april 2024. 

*Size*: 189 songs

*Licence*: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

**Use:**

This dataset can work for **Music Generation** projects using machine learning models such as CNNs, LSTMs or GANs. For the purpose of such a project, *the data does not have to be divided into test and train data*, all of the data will be used for training and hyperparameter adjustments will be done after qualitative (or even quantitative) analysis of the generated music.

**About the format:**

While more computational intensive than MIDI files, using audio files result in more nuanced understanding of audio properties from the model. These will have to be processed into either:
- Waveform Samples.
- Short-Time Fourier Transform (STFT) Spectogram.
- Mel-Spectrogram (Mel scale).
- MFCCs (Mel-Frequency Cepstral Coefficients) for MEL-scale cepstral representation.

I provide a [python script](https://github.com/monica-ayala/AudioGeneration/blob/main/scripts/preprocessing.py) to do the transformation into mel spectogram format using the [librosa](https://pypi.org/project/librosa/) library and save the visual representation into the folder ```/spectogram_images```

For example, these are some of the spectograms:
![image](https://github.com/monica-ayala/AudioGeneration/assets/75228128/d1e2afdf-b67e-4efc-872f-eafad0077241)
![image](https://github.com/monica-ayala/AudioGeneration/assets/75228128/8b28371c-c98c-4414-8531-2c40096dd3b7)
![image](https://github.com/monica-ayala/AudioGeneration/assets/75228128/045c38a6-88bf-4054-b008-ccd1bb39126e)

To mantain the same shape of data (that is, that the shape of each spectogram is consistent) I decided to divde each audio file into 30 seconds each and generate their corresponding spectograms. This gave me a total of 1515 images in the ```/spectogram_images``` folder

**Preprocessing**

After generating the mel spectograms and saving the information into numpy list, we apply the following function to normalize the information in a scale of [0-1]

```
def normalize_spectrogram(spectrogram):
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    return (spectrogram - min_val) / (max_val - min_val)
```

**Data Augmentation**

For data augmentation we can shift the pitch of the spectogram or add random noise to it.

```
def pitch_shift_spectrogram(spectrogram, sample_rate, n_steps):
    audio = librosa.feature.inverse.mel_to_audio(spectrogram, sr=sample_rate, n_fft=2048, hop_length=512)
    audio_shifted = librosa.effects.pitch_shift(audio, sample_rate, n_steps)
    spectrogram_shifted = librosa.feature.melspectrogram(y=audio_shifted, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128)
    return librosa.power_to_db(spectrogram_shifted, ref=np.max)

def add_noise_spectrogram(spectrogram, noise_level=0.005):
    noise = np.random.randn(*spectrogram.shape) * noise_level
    return spectrogram + noise
```

**References**

Briot, JP., Pachet, F. Deep learning for music generation: challenges and directions. Neural Comput & Applic 32, 981–993 (2020). https://doi.org/10.1007/s00521-018-3813-6

**Previous Work**

LSTM Model for music generation using MIDI files: [Github Repository](https://github.com/monica-ayala/MusicGenerator) and [Presentation](https://www.canva.com/design/DAF54orkKw4/GHiqPZIscVxblPPqpttnww/view?utm_content=DAF54orkKw4&utm_campaign=designshare&utm_medium=link&utm_source=editor)

*Note: The dataset I used is different in every possible way from this one, even format, and the model I intend on using will also be different*

