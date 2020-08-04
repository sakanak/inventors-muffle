import pydub
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt

path = "../Audio Recordings/Banana/BananaU1.mp3"
frq_cutoff = 20

# decompress mp3 to temp.wav for processing
input_file = pydub.AudioSegment.from_mp3(path)
input_file.export("temp.wav", format = "wav")
rate, aud_data = wavfile.read("temp.wav")

def frequency_spectrum(x, sf):
    """
    Derive frequency spectrum of a signal from time domain
    :param x: signal in the time domain
    :param sf: sampling frequency
    :returns frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering

    n = len(x)
    # print(n)
    k = np.arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fft.fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)

# pull frequencies and their distributions from function
frq, dist = frequency_spectrum(aud_data, rate)

# find most common frequency range
mini = len(dist)
maxi = 0
for x in range(len(dist)):
    if dist[x] > frq_cutoff:
        if x > maxi:
            maxi = x
        if x < mini:
            mini = x

# pass frequency range into band-pass butterworth filter
a, b = butter(4, (frq[mini] * 2 / rate, frq[maxi] * 2 / rate), "bandpass")

# apply filter to original audio
aud_output = lfilter(a, b, aud_data)

# visualization of freqs/distributions
y = aud_data
t = np.arange(len(y)) / float(rate)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.plot(t, aud_output)
plt.show()

wavfile.write("output.wav", rate, aud_data)