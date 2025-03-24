import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# Loading the signal in librosa
audio = r"/Users/julien/Desktop/ESILV/A3/Dorset/Rapports/DSP Group Project/coyotes.mp3"
signal, sampling_rate = librosa.load(path=audio)

# Printing the waveform
plt.figure(figsize=(12, 4))
librosa.display.waveshow(signal, sr=sampling_rate)
plt.title('Coyote Howl Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Getting some details of the signal
print(f"Sample rate: {sampling_rate} Hz")
print(f"Audio duration: {len(signal)/sampling_rate} seconds")
print(f"Maximum amplitude: {np.max(signal):.4f}")
print(f"Minimum amplitude: {np.min(signal):.4f}")
print(f"Mean amplitude: {np.mean(signal):.4f}")
print(f"Standard deviation: {np.std(signal):.4f}")

# Short Time Fourier Transform
stft = librosa.stft(signal)
signal_dB = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

plt.figure(figsize=(12, 4))
librosa.display.specshow(signal_dB, sr=sampling_rate, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of Coyote Howl')
plt.show()

# Discrete Fourier Transform
N = len(signal)
X = np.fft.fft(signal)
freq = np.fft.fftfreq(N, d=1/sampling_rate)

plt.figure(figsize=(16, 6))
plt.plot(freq[:N//2], np.abs(X[:N//2]))
plt.title('Magnitude Spectrum of Coyote Howl')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.minorticks_on()
plt.show()

# Defining the cut rates
lowcut = 500.0
highcut = 2000.0

nyquist = 0.5 * sampling_rate
low = lowcut / nyquist
high = highcut / nyquist

# FIR filter designing
b = scipy.signal.firwin(numtaps=1001, cutoff=[low, high], pass_zero=False)

# Application of the filter to the signal (convolution)
y_filtered = scipy.signal.lfilter(b, 1, signal)

# Printing the signal before and after the filtering
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
librosa.display.waveshow(signal, sr=sampling_rate)
plt.title('Original Waveform')

plt.subplot(2, 1, 2)
librosa.display.waveshow(y_filtered, sr=sampling_rate)
plt.title('Filtered Waveform')

plt.tight_layout()
plt.show()

# Spectrogram visualisation after the filtering
D_filtered = librosa.stft(y_filtered)
S_db_filtered = librosa.amplitude_to_db(np.abs(D_filtered), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db_filtered, sr=sampling_rate, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Filtered Spectrogram')
plt.show()