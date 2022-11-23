import matplotlib.pyplot as plot
import numpy as np
import pyaudio
import signal
import torch
import torchcrepe


CP = 440
C0 = 2 ** (-(9 + 4*12) / 12)

run = True


def scale():

    return ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def note(semitone):

    return scale()[semitone % 12]


def semitone(frequency, cp=CP, c0=C0):

    return round(12 * np.log2(frequency / (cp * c0)))


def a_weighting(f):

    a = f**4 * 12194**2
    b = f**2 + 20.6**2
    c = f**2 + 107.7**2
    d = f**2 + 737.9**2
    e = f**2 + 12194**2

    w = a / (b * np.sqrt(c * d) * e)

    return w


def a_weighting_db(f):

    w = a_weighting(f) + 1e-7
    w = 20 * np.log10(w) + 2

    return w


def stop(*args, **kwargs):

    global run
    run = False


def start():

    sr = torchcrepe.SAMPLE_RATE
    ws = torchcrepe.WINDOW_SIZE

    args = dict(model='full', sample_rate=sr)
    roi = dict(fmin=50, fmax=1000)
    hop = dict(hop_length=int(10e-3 * sr))

    w = np.hanning(ws)
    f = np.fft.rfftfreq(ws, 1/sr)
    a = a_weighting(f)
    adb = a_weighting_db(f)

    audio = pyaudio.PyAudio()

    stream = audio.open(
        input=True,
        input_device_index=None,
        rate=sr,
        channels=1,
        format=pyaudio.paFloat32)

    while run:

        x = stream.read(ws, exception_on_overflow=False)
        x = np.frombuffer(x, dtype=np.float32)

        xt = torch.tensor(x)[None]
        yt = torchcrepe.predict(xt, **args, **roi, **hop)

        y = yt.detach().numpy().astype(float)
        y = np.mean(y)

        z = np.fft.rfft(w * x, norm='forward')
        z = np.abs(z)

        zdb = 20 * np.log10(z + 1e-7)
        zpeak = 20 * np.log10(np.max(z * a) + 1e-7)

        n = note(semitone(y)).ljust(2)

        plot.clf()

        plot.axvline(x=roi['fmin'], color='red', linestyle='dashed', label='f-range')
        plot.axvline(x=roi['fmax'], color='red', linestyle='dashed')

        plot.plot(f, zdb + adb, color='blue', linestyle='dashed', label='a-weighting')
        plot.plot(f, zdb, color='black', label='spectrum')

        plot.axhline(y=zpeak, color='blue', label='loudness')
        plot.axvline(x=y, color='red', label='estimate')

        plot.title(n)
        plot.legend()

        plot.xscale('log')
        plot.ylim(-120, 0)

        plot.draw()
        plot.pause(100e-3)

    stream.close()
    audio.terminate()


if __name__ == '__main__':

    signal.signal(signal.SIGINT, stop)
    plot.figure(f'A4 = {CP} Hz').canvas.mpl_connect('close_event', stop)

    start()
