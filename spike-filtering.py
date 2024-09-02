import matplotlib.pyplot as plt
import numpy as np
from if_neuron import IF_neuron
from numpy.typing import ArrayLike
from nengo.synapses import Lowpass


def lowpass_filter(x: ArrayLike, sample_freq: int = 1000, cutoff_freq: int = 100):
    # FFT of the signal
    freq_domain = np.fft.fft(x)

    # Create a low-pass filter
    freqs = np.fft.fftfreq(len(x), d=1 / sample_freq)
    low_pass_filter = np.abs(freqs) < cutoff_freq

    # Apply the filter in the frequency domain
    filtered_freq_domain = freq_domain * low_pass_filter

    # Inverse FFT to get back to the time domain
    return np.fft.ifft(filtered_freq_domain)


DURATION = 10
DT = 0.1
N_STEPS = int(DURATION / DT)
timesteps = np.arange(0, DURATION, DT)

inp = np.sin(timesteps)
spikes = IF_neuron.spike_input(inp)
# Test nengo filters for comparisons
tau = 1e-3
syn = Lowpass(tau)
filtered_nengo = syn.filt(spikes)
filtered = lowpass_filter(spikes)

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
