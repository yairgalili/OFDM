import itertools

import numpy as np
import matplotlib.pyplot as plt

from ex3 import qpsk_mod
from ex4 import simulate_sm_qpsk

# -------------------------------------------
# Spatial Multiplexing Zero Forcing (Matrix Form)
# -------------------------------------------

def simulate_sm_svd(snr_db_values, Nrx, Ntx, num_symbols=100000):
    ser = []
    
    # Generate random QPSK bits and symbols (matrix form)
    bits = np.random.randint(0, 2, (2, num_symbols, Ntx, 1))
    s = qpsk_mod(bits)
    # shape (N, Ntx, 1)
        
    for snr_db in snr_db_values:
        snr = 10**(snr_db / 10)
        noise_var = 1 / (2 * snr)   # QPSK normalization
        
        # Rayleigh fading (matrix form)
        H = (np.random.randn(num_symbols, Nrx, Ntx)+ 1j*np.random.randn(num_symbols, Nrx, Ntx)) / np.sqrt(2)
        
        # Noise
        n = (np.random.randn(num_symbols, Nrx, 1) + 1j*np.random.randn(num_symbols, Nrx, 1)) * np.sqrt(noise_var)
        
        _, _, Vh = np.linalg.svd(H, full_matrices=False)
        # (N, Nrx, 1)
        
        x = np.conj(Vh).transpose((0, 2, 1)) @ s/np.sqrt(Ntx)
        y = H@x + n
        # (N, Ntx, Nrx) @ (N,  Nrx, 1) = (N, Ntx, 1)
        r_comb = np.linalg.pinv(H) @ y
        
        # QPSK detection (slicing real & imag signs)
        
        s_hat = np.sign(np.real(r_comb)) + 1j*np.sign(np.imag(r_comb))
        s_hat = s_hat / np.sqrt(2)
                
        # Symbol errors
        errors = np.sum(s_hat != s)
        ser.append(errors / s.size)
        
    return ser

if __name__ == "__main__":
	# -------------------------------------------
	# Run Simulation
	# -------------------------------------------
	SNR_dB = np.arange(0, 21, 2)
	
	results_zf_22 = simulate_sm_svd(SNR_dB, 2, 2)
	
	# -------------------------------------------
	# Plot
	# -------------------------------------------
	plt.figure(figsize=(7,5))
	plt.semilogy(SNR_dB, results_sm_22, marker='o', label=f"SM ML 2×2")
	plt.semilogy(SNR_dB, results_sm_42, marker='o', label=f"SM ML 4×2")
	plt.semilogy(SNR_dB, results_zf_22, marker='o', label=f"SM ZF 2×2")
	plt.semilogy(SNR_dB, results_zf_42, marker='o', label=f"SM ZF 4×2")
	plt.grid(True, which="both", linestyle='--', alpha=0.7)
	plt.xlabel("SNR[dB]")
	plt.ylabel("symbol error rate (SER)")
	plt.legend()
	plt.savefig("ex5.jpg")