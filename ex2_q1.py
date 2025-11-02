import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------
# QPSK + Rayleigh + MRC (Matrix Form)
# -------------------------------------------


def qpsk_mod(bits):
    """
    Gray-coded QPSK mapping.
    bits: shape (2, num_symbols)
    """
    b0 = bits[0]
    b1 = bits[1]
    # Map bits to QPSK symbols
    # (00)-> +1+1j, (01)-> +1-1j, (11)-> -1-1j, (10)-> -1+1j
    symbols = (1 - 2 * b0) + 1j * (1 - 2 * b1)
    # Normalize to unit power
    return symbols / np.sqrt(2)


def simulate_mrc_qpsk(N_rx, snr_db_values, num_symbols=200000):
    ser = []

    # Generate random QPSK bits and symbols (matrix form)
    bits = np.random.randint(0, 2, (2, num_symbols))  # shape (2, N)
    s = qpsk_mod(bits)  # shape (N,)

    # Repeat across branches: shape (N_rx, N)
    s_rep = np.tile(s, (N_rx, 1))

    for snr_db in snr_db_values:
        snr = 10 ** (snr_db / 10)
        noise_var = 1 / (2 * snr)  # QPSK normalization

        # Rayleigh fading (matrix form)
        h = (
            np.random.randn(N_rx, num_symbols) + 1j * np.random.randn(N_rx, num_symbols)
        ) / np.sqrt(2)

        # Noise
        n = (
            np.random.randn(N_rx, num_symbols) + 1j * np.random.randn(N_rx, num_symbols)
        ) * np.sqrt(noise_var)

        # Received signal
        r = h * s_rep + n

        # MRC combining (vectorized)
        r_comb = np.sum(np.conjugate(h) * r, axis=0)

        # QPSK detection (slicing real & imag signs)
        s_hat = np.sign(np.real(r_comb)) + 1j * np.sign(np.imag(r_comb))
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
    N_rx_list = [2, 4]

    results = {}
    for N in N_rx_list:
        results[N] = simulate_mrc_qpsk(N, SNR_dB)

    # -------------------------------------------
    # Plot
    # -------------------------------------------
    plt.figure(figsize=(7, 5))
    for N in N_rx_list:
        plt.semilogy(SNR_dB, results[N], marker="o", label=f"MRC 1×{N}")

    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.xlabel("SNR[dB]")
    plt.ylabel("symbol error rate (SER)")
    plt.grid(True)
    plt.legend()
    plt.savefig("ex2_q1.jpg")
