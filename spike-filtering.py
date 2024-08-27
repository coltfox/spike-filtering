import matplotlib.pyplot as plt
import numpy as np
from if_neuron import IF_neuron
from numpy.typing import ArrayLike
from nengo.synapses import Lowpass
from scipy import signal


def generate_spikes(n_steps: int, freq: int = 0.1, amplitude: float = 1):
    """
    Generate spikes give an frequency

    Args:
        n_steps (int): Number of time steps
        freq (int): Frequency of spikes
        amplitude (float): Amplitude of spikes
    """
    spikes = np.zeros((n_steps,))
    # Time steps between each spike
    ts = freq * n_steps

    for t in range(n_steps):
        if t % ts != 0:
            continue

        spikes[t] = amplitude

    return spikes

# f(t) = (1 / tau) * exp(-t / tau)


def filter_spikes(spikes: ArrayLike):
    # Source: https://lonewritings.github.io/2018/05/26/signal_filtering.html
    # sampling rate defines number of observations in one second.
    sampling_rate = 4000

    # Suppose the cut-off frequency is 100
    fc = 1000
    # Design of digital filter requires cut-off frequency to be normalised by sampling_rate/2
    w = fc / (sampling_rate / 2)
    num, den = signal.butter(5, w, 'low', analog=False)
    return signal.filtfilt(num, den, spikes)
    # return signal.lfilter(num, den, spikes)


def lowpass_filter(x: ArrayLike, t: ArrayLike, tau: float = 1e-3):
    # tau = max(1 - synapse, )
    # Source: https://github.com/nengo/nengo/blob/30f711c26479a94e486ab1862a1400dce5b3ffa0/nengo/synapses.py#L436
    filt = (1 / tau) * np.exp(-t / tau)
    return np.convolve(x, filt, mode='same')


DURATION = 10
DT = 0.1
N_STEPS = int(DURATION / DT)
timesteps = np.arange(0, DURATION, DT)

inp = np.sin(timesteps)
spikes = IF_neuron.spike_input(inp)
# Test nengo filters for comparions
tau = 1e-3
syn = Lowpass(tau)
filtered_nengo = syn.filt(spikes)
# filtered = lowpass_filter(spikes, timesteps, tau=1 - tau)
filtered = filter_spikes(spikes)

plt.figure()
plt.xlabel('t')
plt.ylabel('sin(t)')
plt.title('Input Signal')
plt.grid(True)
plt.plot(timesteps, inp)

plt.figure()
plt.xlabel('t')
plt.ylabel('S')
plt.title('Spikes')
plt.grid(True)
plt.plot(timesteps, spikes)

plt.figure()
plt.xlabel('t')
plt.ylabel('S')
plt.title('Filtered Spikes')
plt.grid(True)
plt.plot(timesteps, filtered)

plt.figure()
plt.xlabel('t')
plt.ylabel('S')
plt.title('Filtered Spikes Nengo')
plt.grid(True)
plt.plot(timesteps, filtered_nengo)

plt.show()
