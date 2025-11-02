"""

Repeat the SISO case (Tx, channel, noise addition, Rx) and produce the SER plots (Rayleigh and AWGN)
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# QPSK Modulation / Demodulation
# ---------------------------------------------------------
def qpsk_mod(bits):
    b0 = bits[:, 0]
    b1 = bits[:, 1]
    symbols = ((2 * b0 - 1) + 1j * (2 * b1 - 1)) / np.sqrt(2)
    return symbols


def qpsk_demod(symbols):
    real = (np.real(symbols) >= 0).astype(int)
    imag = (np.imag(symbols) >= 0).astype(int)
    return np.vstack((real, imag)).T


# ---------------------------------------------------------
# SER Simulation
# ---------------------------------------------------------
def simulate_ser(snr_db, num_symbols=100000, fading=False):
    k = 2  # bits per QPSK symbol
    bits = np.random.randint(0, 2, (num_symbols, k))
    s = qpsk_mod(bits)  # QPSK symbols

    Es = 1
    snr_linear = 10 ** (snr_db / 10)
    No = Es / snr_linear
    noise_std = np.sqrt(No / 2)

    noise = noise_std * (
        np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)
    )

    if fading:
        h = (
            np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)
        ) / np.sqrt(2)
        y = h * s + noise
        r = y / h  # Perfect CSI equalization
    else:
        r = s + noise

    detected_bits = qpsk_demod(r)
    symbol_errors = np.any(detected_bits != bits, axis=1)
    ser = np.mean(symbol_errors)
    return ser


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Run Simulation
    # ---------------------------------------------------------
    snr_db_range = np.arange(0, 21, 2)
    ser_awgn = []
    ser_rayleigh = []

    for snr in snr_db_range:
        ser_awgn.append(simulate_ser(snr, fading=False))
        ser_rayleigh.append(simulate_ser(snr, fading=True))
        print(
            f"SNR={snr:2d} dB -> AWGN SER={ser_awgn[-1]:.4e}, Rayleigh SER={ser_rayleigh[-1]:.4e}"
        )

    # Convert SNR to Eb/N0 (since QPSK has 2 bits/symbol)
    ebn0_db = snr_db_range - 10 * np.log10(2)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.semilogy(ebn0_db, ser_awgn, "o-", label="AWGN")
    plt.semilogy(ebn0_db, ser_rayleigh, "s--", label="Rayleigh Fading (Equalized)")

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.title("SISO QPSK SER Performance")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.ylim([1e-5, 1])
    plt.savefig("ex1_SISO.jpg")
