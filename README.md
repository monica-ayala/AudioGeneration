# AudioGeneration
---
**TC3002B: Desarrollo de aplicaciones avanzadas de ciencias computacionales (Gpo 201)**

**Mónica Andrea Ayala Marrero - A01707439**

---
**About the dataset:**

This dataset contains 188 songs from Taylor Swift's discography in mp3 format. The songs were uploaded by different users to the https://archive.org/ platform and compiled manually by me.

**Sources for each album:**
- ***Taylor Swift*** [[link]](https://archive.org/details/cd_taylor-swift_taylor-swift/disc1/01.+Taylor+Swift+-+Tim+McGraw.flac) | 10 songs
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

This dataset can work for **Music Generation** projects using machine learning models such as VAEs, LSTMs or GANs. For the purpose of such a project, *the data does not have to be divided into test and train data*, all of the data will be used for training and hyperparameter adjustments will be done after qualitative (or even quantitative) analysis of the generated music.

More precisely, I will be working with a **Variational Autoencoder Model** that intakes spectogram data of the audio files and outputs reconstructed spectogram data. After training such model I will use the *decoder* to generate new spectogram data from samples of the ```prior``` distribution. I will then transform the spectogram into an audio format such as wav or mp3.

**About the format:**

While more computational intensive than MIDI files, using audio files result in more nuanced understanding of audio properties from the model. These will have to be processed into either:
- Waveform Samples.
- Short-Time Fourier Transform (STFT) Spectogram.
- Mel-Spectrogram (Mel scale).
- MFCCs (Mel-Frequency Cepstral Coefficients) for MEL-scale cepstral representation.

I provide two python scripts, [the first one](https://github.com/monica-ayala/AudioGeneration/blob/main/scripts/mel-preprocessing.py) to do the transformation from mp3 into mel spectogram format using the [librosa](https://pypi.org/project/librosa/) library and save the visual representation into the folder ```/spectogram_images```

For example, these are some of the spectograms:
![image](https://github.com/monica-ayala/AudioGeneration/assets/75228128/d1e2afdf-b67e-4efc-872f-eafad0077241)
![image](https://github.com/monica-ayala/AudioGeneration/assets/75228128/8b28371c-c98c-4414-8531-2c40096dd3b7)
![image](https://github.com/monica-ayala/AudioGeneration/assets/75228128/045c38a6-88bf-4054-b008-ccd1bb39126e)

And [the second one](https://github.com/monica-ayala/AudioGeneration/blob/main/scripts/stft-preprocessing.py) to do the transformation into a Short-Time Fourier Transform (STFT) Spectogram and saving the visual representations to ```/stft_images``` and the data into ```stft_spectogram.npz```

To mantain the same shape of data (that is, that the shape of each spectogram is consistent) I decided to divde each audio file into 30 seconds each and generate their corresponding spectograms. This gave me a total of 1515 spectograms of which I decided to prune out all of those that didn't complete 30 seconds, resulting in **1507** spectograms.

**Preprocessing**

I decided to use the STFT Spectograms as they provide easier reconstrucion from STFT spectogram to .wav format. In ```/scripts/stft_preprocessing.py``` I use the following functions to normalize the spectogram (get the data to be from [0-1]) and then to reshape each spectogram from shape (1025, 2584) to (1024, 2048,1) so that my VAE Model can better downscale and upscale the data.

```
def normalize_spectrogram(spectrogram):
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    normalized = (spectrogram - min_val) / (max_val - min_val)
    return normalized

def reshape(spectogram):
    if spectogram.shape >= (1024, 2048):
        reshaped = spectogram[:1024, :2048, np.newaxis] 
    else:
        reshaped = np.zeros((1024, 2048, 1))
        reshaped[:spectogram.shape[0], :spectogram.shape[1], 0] = spectogram

    return reshaped
```
**Current Issues**
- I cannot undo the normalization of the spectograms as I did not save their min_val and max_val, which in turn hinders the audio reconstruction.
- The encoder is defined with both MaxPooling2D and Conv2D layers with 'agresive' strides that perhaps reduce the dimensionality too quickly.
- The decoder upscales the data in a disimilar manner to that in which the encoder downscales it.

**References**

Briot, JP., Pachet, F. Deep learning for music generation: challenges and directions. Neural Comput & Applic 32, 981–993 (2020). https://doi.org/10.1007/s00521-018-3813-6

**Previous Work**

LSTM Model for music generation using MIDI files: [Github Repository](https://github.com/monica-ayala/MusicGenerator) and [Presentation](https://www.canva.com/design/DAF54orkKw4/GHiqPZIscVxblPPqpttnww/view?utm_content=DAF54orkKw4&utm_campaign=designshare&utm_medium=link&utm_source=editor)

*Note: The dataset I used is different in every possible way from this one, even format, and the model I intend on using will also be different*

