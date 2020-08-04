import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt

words = ("Banana", "Cellphone", "Corona", "Exterior", "FaceMask", "Hello", "Orange", "Quantify", "Zebra")
states = ("M", "U")

for word in words:
    for state in states:
        for x in range(1, 6):
            # load in audio sample
            audio = tfio.audio.AudioIOTensor("../Audio Recordings/{0}/{0}{1}{2}.mp3".format(word, state, x))

            # remove last dimension
            audio_tensor = tf.squeeze(audio.to_tensor(), axis=[-1])

            # Convert to spectrogram
            spectrogram = tfio.experimental.audio.spectrogram(
                audio_tensor, nfft=2048, window=1024, stride=512)

            plt.figure()
            plt.xlabel("Windows")
            plt.ylabel("Frequency")
            plt.imshow(tf.math.log(spectrogram).numpy())
            plt.savefig("{0}/{0}{1}{2}.png".format(word, state, x))