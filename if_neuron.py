import numpy as np


class IF_neuron(object):
    def __init__(self, v_thr=1.0):
        """
        Args:
          v_thr <int>: Threshold voltage of the IF neuron, default 1.0 .
        """
        self._v_thr = v_thr
        self._v = 0

    @staticmethod
    def spike_input(inp, synapse=0.01):
        if_neuron = IF_neuron(v_thr=1)  # Instantiate a neuron.
        T = inp.shape[0]

        spikes = []
        # Execute the IF neuron for the entire duration of the input.
        for t in range(T):
            spikes.append(if_neuron.encode_input_and_spike(inp[t], synapse))

        return np.array(spikes)

    def encode_input_and_spike(self, inp, synapse=0.01):
        """
        Integrates the input and produces a spike if the IF neuron's voltage reaches 
        or crosses threshold.

        Args:
          inp <float>: Scalar input to the IF neuron.
        """
        self._v = self._v + inp
        if self._v >= self._v_thr:
            self._v = 0  # Reset the voltage and produce a spike.
            return 1.0  # Spike.
        elif self._v < 0:  # Rectify the voltage if it is negative.
            self._v = 0

        return 0  # No spike.
