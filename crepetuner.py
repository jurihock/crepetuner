import click
import matplotlib
import matplotlib.pyplot as plot
import numpy as np
import pyaudio
import signal
import torch
import torchcrepe


matplotlib.rcParams['axes.titlesize'] = 'xx-large'
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['axes.titlepad'] *= 3

CP = 440
C0 = 2 ** (-(9 + 4*12) / 12)

run = False


def scale():

    return ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def note(semitone):

    return scale()[semitone % 12]


def semitone(frequency, cp=CP, c0=C0):

    return round(12 * np.log2(frequency / (cp * c0)))


def dbfs(value, bias=1e-7):

    return 20 * np.log10(value + bias)


def a_weighting(frequency):

    f4 = frequency**4
    f2 = frequency**2

    a = f4 * 12194**2
    b = f2 + 20.6**2
    c = f2 + 107.7**2
    d = f2 + 737.9**2
    e = f2 + 12194**2

    return a / (b * np.sqrt(c * d) * e)


def a_weighting_db(frequency):

    return dbfs(a_weighting(frequency)) + 2


def stop(*args, **kwargs):

    global run
    run = False


def start(device, cp, roi, model, pause, style):

    global run
    run = True

    sr = torchcrepe.SAMPLE_RATE
    ws = torchcrepe.WINDOW_SIZE

    args = dict(model=model, sample_rate=sr)
    roi = dict(fmin=min(roi), fmax=max(roi))
    hop = dict(hop_length=int(10e-3 * sr))

    w = np.hanning(ws)
    f = np.fft.rfftfreq(ws, 1/sr)
    a = a_weighting(f)
    adb = a_weighting_db(f)

    audio = pyaudio.PyAudio()

    stream = audio.open(
        input=True,
        input_device_index=device,
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

        zdb = dbfs(z)
        zpeak = dbfs(np.max(z * a))

        n = note(semitone(y, cp)).ljust(2)

        plot.clf()

        plot.axvline(x=roi['fmin'], color='tab:red', linestyle='dashed', label='f-range')
        plot.axvline(x=roi['fmax'], color='tab:red', linestyle='dashed')

        plot.plot(f, zdb + adb, color='tab:blue', linestyle='dashed', label='a-weighting')
        plot.plot(f, zdb, color=['black', 'white'][style], label='spectrum')

        plot.axhline(y=zpeak, color='tab:blue', label='loudness')
        plot.axvline(x=y, color='tab:red', label='estimate')

        plot.title(n)
        plot.legend()

        plot.xlabel('Hz')
        plot.ylabel('dB')

        plot.xscale('log')
        plot.ylim(-120, 0)

        plot.draw()
        plot.pause(pause * 1e-3)

    stream.stop_stream()
    stream.close()
    audio.terminate()


def probe():

    audio = pyaudio.PyAudio()
    devices = audio.get_device_count()

    for i in range(devices):

        device = audio.get_device_info_by_index(i)

        print(device.get('name'))


def find(source):

    if not source:
        return None

    name = source.lower()

    audio = pyaudio.PyAudio()
    devices = audio.get_device_count()

    for i in range(devices):

        device = audio.get_device_info_by_index(i)

        if name in device.get('name').lower():
            return i

    return None


@click.command('crepetuner', help='CREPE tuner, not turner', context_settings=dict(max_content_width=100, help_option_names=['-h', '--help']))
@click.argument('source', type=str, required=False)
@click.option('-a', '--a4', default=440, type=int, show_default=True, help='Concert pitch in hertz.')
@click.option('-f', '--freqs', default=(50, 1000), type=(int, int), show_default=True, help='Frequency range in hertz.')
@click.option('-m', '--model', default='full', type=str, show_default=True, help='Model name tiny or full.')
@click.option('-p', '--pause', default=1, type=int, show_default=True, help='Delay in milliseconds.')
@click.option('-s', '--style', default='black', type=str, show_default=True, help='Style black or white.')
@click.option('-l', '--list', is_flag=True, default=False, help='List available audio sources.')
def main(source, a4, freqs, model, pause, style, list):

    if list:
        probe()
        exit()

    device = find(source)

    style = ['white', 'black'].index(style)

    if style:
        plot.style.use('dark_background')

    signal.signal(signal.SIGINT, stop)

    figure = plot.figure(f'A4 = {a4} Hz')
    figure.canvas.mpl_connect('close_event', stop)

    start(device=device,
          cp=a4,
          roi=freqs,
          model=model,
          pause=pause,
          style=style)


if __name__ == '__main__':

    main()
