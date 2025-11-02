import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------
# Eigen BF (Matrix Form)
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

def simulate_eigen_bf_qpsk(snr_db_values, Nrx, Ntx, num_symbols=1000000):
    ser = []
    
    # Generate random QPSK bits and symbols (matrix form)
    bits = np.random.randint(0, 2, (2, num_symbols))
    s = qpsk_mod(bits)                                     # shape (N)
        
    for snr_db in snr_db_values:
        snr = 10**(snr_db / 10)
        noise_var = 1 / (2 * snr)   # QPSK normalization
        
        # Rayleigh fading (matrix form)
        H = (np.random.randn(num_symbols, Nrx, Ntx)+ 1j*np.random.randn(num_symbols, Nrx, Ntx)) / np.sqrt(2)
        
        # Noise
        n = (np.random.randn(num_symbols, Nrx) + 1j*np.random.randn(num_symbols, Nrx)) * np.sqrt(noise_var)
        
        # Received signal
        _, _, w = np.linalg.svd(H, full_matrices=False)
        h = np.squeeze(H@np.expand_dims(w[:, 0, :].conj(), axis=2))
        r = h*np.expand_dims(s, axis=1) + n
        
        r_comb = np.sum(np.conj(h) * r, axis=1)
        
        # QPSK detection (slicing real & imag signs)
        s_hat = np.sign(np.real(r_comb)) + 1j*np.sign(np.imag(r_comb))
        s_hat = s_hat / np.sqrt(2)
        
        # Symbol errors
        errors = np.sum(s_hat != s)
        ser.append(errors / num_symbols)
        
    return ser

def simulate_stc_qpsk(snr_db_values, Nrx, num_symbols=1000000):
    ser = []
    T = 2 # two time slots
    Ntx = T
    # Generate random QPSK bits and symbols (matrix form)
    bits = np.random.randint(0, 2, (2, T, 1, num_symbols))       # shape (2, T, N)
    s = qpsk_mod(bits)
    # shape (T, 1, N), two time slots
        
    for snr_db in snr_db_values:
        snr = 10**(snr_db / 10)
        noise_var = 1 / (2 * snr)   # QPSK normalization
        
        # Rayleigh fading (matrix form)
        h = (np.random.randn(Ntx, Nrx, num_symbols) + 1j*np.random.randn(Ntx, Nrx, num_symbols)) / np.sqrt(2)
        
        # Noise
        n = (np.random.randn(T, Nrx, num_symbols) + 1j*np.random.randn(T, Nrx, num_symbols)) * np.sqrt(noise_var)
        
        # Received signal
        # (1, N)*(Nrx, N) + (Nrx, N)
        r0 = s[0]*h[0] + s[1]*h[1] + n[0]
        r1 = -np.conj(s[1])*h[0] + np.conj(s[0])*h[1] + n[1]
        
        # STC (vectorized), normalization does not affect on estimation in QPSK
        # (Nrx, N) * (Nrx, N)
        s0_hat = r0 * np.conj(h[0]) + np.conj(r1) * h[1]
        s1_hat = r0 * np.conj(h[1]) - np.conj(r1) * h[0]
        # (2, Nrx, N)
        r_comb = np.sum(np.stack((s0_hat, s1_hat), axis=0), axis=1)
        
        # QPSK detection (slicing real & imag signs)
        s_hat = np.sign(np.real(r_comb)) + 1j*np.sign(np.imag(r_comb))
        s_hat = s_hat / np.sqrt(2)
        
        # Symbol errors
        errors = np.sum(s_hat != np.squeeze(s))
        ser.append(errors / num_symbols)
        
    return ser

# -------------------------------------------
# Run Simulation
# -------------------------------------------
SNR_dB = np.arange(0, 21, 2)

results_bf_22 = simulate_eigen_bf_qpsk(SNR_dB, 2, 2)
results_bf_24 = simulate_eigen_bf_qpsk(SNR_dB, 2, 4)
results_stc = simulate_stc_qpsk(SNR_dB, 2)
results_stc_4 = simulate_stc_qpsk(SNR_dB, 4)
# -------------------------------------------
# Plot
# -------------------------------------------
plt.figure(figsize=(7,5))
plt.semilogy(SNR_dB, results_bf_22, marker='o', label=f"BF 2×2")
plt.semilogy(SNR_dB, results_bf_24, marker='o', label=f"BF 2×4")
plt.semilogy(SNR_dB, results_stc, marker='o', label=f"STC 2×2")
plt.semilogy(SNR_dB, results_stc_4, marker='o', label=f"STC 2×4")

plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.xlabel("SNR[dB]")
plt.ylabel("symbol error rate (SER)")
plt.legend()
plt.savefig("ex3.jpg")
