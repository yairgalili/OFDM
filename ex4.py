import itertools

import numpy as np
import matplotlib.pyplot as plt

from ex3 import simulate_stc_qpsk
# -------------------------------------------
# Spatial Multiplexing (Matrix Form)
# -------------------------------------------

def qpsk_mod(bits):
    """
    Gray-coded QPSK mapping.
    bits: shape (2, ...)
    """
    b0 = bits[0]
    b1 = bits[1]
    # Map bits to QPSK symbols
    # (00)-> +1+1j, (01)-> +1-1j, (11)-> -1-1j, (10)-> -1+1j
    symbols = (1 - 2*b0) + 1j*(1 - 2*b1)
    # Normalize to unit power
    return symbols / np.sqrt(2)

def generate_all_qpsk_combinations(sequence_length):
    """
    Generates all possible combinations of QPSK symbol sequences of a given length.

    :param sequence_length: The desired length of the symbol sequence.
    :return: A list of tuples, where each tuple is a unique QPSK symbol sequence.
    """
    # Use itertools.product to get the Cartesian product of the qpsk_points 
    # repeated 'sequence_length' times.
    qpsk_points = np.array([1+1j, 1-1j, -1+1j, -1-1j])/np.sqrt(2)
    all_combinations = list(itertools.product(qpsk_points, repeat=sequence_length))
    return np.array(all_combinations)


def simulate_sm_qpsk(snr_db_values, Nrx, Ntx, num_symbols=10000):
    ser = []
    # (dict_size=4^Ntx, Ntx) -> (1, Ntx, dict_size)
    dictionary = generate_all_qpsk_combinations(Ntx).T[np.newaxis, :, :]
    # Generate random QPSK bits and symbols (matrix form)
    bits = np.random.randint(0, 2, (2, num_symbols, Ntx, 1))
    s = qpsk_mod(bits)
    # shape (N, Ntx, 1)
        
    for snr_db in snr_db_values:
        snr = 10**(snr_db / 10)
        noise_var = 1 / (2 * snr)   # QPSK normalization
        
        # Rayleigh fading (matrix form)
        H = (np.random.randn(num_symbols, Nrx, Ntx)+ 1j*np.random.randn(num_symbols, Nrx, Ntx)) / np.sqrt(2*Ntx)
        
        # Noise
        n = (np.random.randn(num_symbols, Nrx, 1) + 1j*np.random.randn(num_symbols, Nrx, 1)) * np.sqrt(noise_var)
        
        # (N, Nrx, 1)
        # y = H@s + n
        y = H@s
        
        # (N, Nrx, 1) - (N, Nrx, Ntx)@(N, Ntx, dict_size) = (N, Nrx, dict_size) -> (N, dict_size)
        r = np.linalg.norm(y - H@dictionary, axis=1)
        
        # QPSK detection (slicing real & imag signs)
        idx = np.argmax(r, axis=1)
        # (1, Ntx, N) -> (N, Ntx, 1)
        s_hat = np.permute_dims(dictionary[:, :, idx], axes=(2, 1, 0))
        
        
        # Symbol errors
        errors = np.sum(s_hat != s) / Ntx
        ser.append(errors / num_symbols)
        
    return ser


# -------------------------------------------
# Run Simulation
# -------------------------------------------
SNR_dB = np.arange(0, 21, 2)

results_sm = simulate_sm_qpsk(SNR_dB, 2, 2)
results_stc = simulate_stc_qpsk(SNR_dB, 1)

# -------------------------------------------
# Plot
# -------------------------------------------
plt.figure(figsize=(7,5))
plt.semilogy(SNR_dB, results_sm, marker='o', label=f"SM 2×2")
plt.semilogy(SNR_dB, results_stc, marker='o', label=f"STC 2×1")

plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.xlabel("SNR[dB]")
plt.ylabel("symbol error rate (SER)")
plt.legend()
plt.savefig("ex4.jpg")