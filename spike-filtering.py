import matplotlib.pyplot as plt
import numpy as np
from if_neuron import IF_neuron
from numpy.typing import ArrayLike
from nengo.synapses import Lowpass

def filter_spikes(spikes, alpha=0.8):
    filt = np.zeros_like(spikes)

    filt[0] = spikes[0]

    for i, spike in enumerate(spikes[1:]):
        filt[i + 1] = alpha * filt[i] + spike
    
    return filt


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
filtered = filter_spikes(spikes)

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Adjust figsize for better spacing

# First plot (top-left)
axs[0, 0].plot(timesteps, inp)
axs[0, 0].set_title('Input Signal')
axs[0, 0].set_xlabel('t')
axs[0, 0].set_ylabel('sin(t)')
axs[0, 0].grid(True)

# Second plot (top-right)
axs[0, 1].plot(timesteps, spikes)
axs[0, 1].set_title('Spikes')
axs[0, 1].set_xlabel('t')
axs[0, 1].set_ylabel('S')
axs[0, 1].grid(True)

# Third plot (bottom-left)
axs[1, 0].plot(timesteps, filtered)
axs[1, 0].set_title('Filtered Spikes')
axs[1, 0].set_xlabel('t')
axs[1, 0].set_ylabel('S')
axs[1, 0].grid(True)

# Fourth plot (bottom-right)
axs[1, 1].plot(timesteps, filtered_nengo)
axs[1, 1].set_title('Filtered Spikes Nengo')
axs[1, 1].set_xlabel('t')
axs[1, 1].set_ylabel('S')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
