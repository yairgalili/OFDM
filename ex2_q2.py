import numpy as np
import matplotlib.pyplot as plt

from ex2_q1 import simulate_mrc_qpsk

# -------------------------------------------
# STC vs MRC (Matrix Form)
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

def simulate_stc_qpsk(snr_db_values, num_symbols=200000):
    ser = []
    T = 2 # two time slots
    Ntx = 2
    # Generate random QPSK bits and symbols (matrix form)
    bits = np.random.randint(0, 2, (2, T, num_symbols))       # shape (2, T, N)
    s = qpsk_mod(bits)                                     # shape (T, N), two time slots
        
    for snr_db in snr_db_values:
        snr = 10**(snr_db / 10)
        noise_var = 1 / (2 * snr)   # QPSK normalization
        
        # Rayleigh fading (matrix form)
        h = (np.random.randn(Ntx, num_symbols) + 1j*np.random.randn(Ntx, num_symbols)) / np.sqrt(2)
        
        # Noise
        n = (np.random.randn(T, num_symbols) + 1j*np.random.randn(T, num_symbols)) * np.sqrt(noise_var)
        
        # Received signal
        
        r0 = s[0]*h[0] + s[1]*h[1] + n[0]
        r1 = -np.conj(s[1])*h[0] + np.conj(s[0])*h[1] + n[1]
        
        # STC (vectorized), normalization does not affect on estimation in QPSK
        s0_hat = r0 * np.conj(h[0]) + np.conj(r1) * h[1]
        s1_hat = r0 * np.conj(h[1]) - np.conj(r1) * h[0]
        r_comb = np.stack((s0_hat, s1_hat), axis=0)
        
        # QPSK detection (slicing real & imag signs)
        s_hat = np.sign(np.real(r_comb)) + 1j*np.sign(np.imag(r_comb))
        s_hat = s_hat / np.sqrt(2)
        
        # Symbol errors
        errors = np.sum(s_hat != s)
        ser.append(errors / num_symbols)
        
    return ser


# -------------------------------------------
# Run Simulation
# -------------------------------------------
SNR_dB = np.arange(0, 21, 2)

results_stc = simulate_stc_qpsk(SNR_dB)
results_mrc = simulate_mrc_qpsk(2, SNR_dB)
# -------------------------------------------
# Plot
# -------------------------------------------
plt.figure(figsize=(7,5))
plt.semilogy(SNR_dB, results_stc, marker='o', label=f"STC 2×1")
plt.semilogy(SNR_dB, results_mrc, marker='o', label=f"MRC 1×2")

plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.xlabel("SNR[dB]")
plt.ylabel("symbol error rate (SER)")
plt.legend()
plt.savefig("ex2_q2.jpg")
