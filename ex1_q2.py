import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Parameters
    f = 2.5e9  # Frequency = 2.5 GHz
    num_paths = 10  # Number of multipath components
    N = 10000  # Number of channel realizations
    tau_min = 50 / (3e8)  # 50 m
    tau_max = 100 / (3e8)  # 100 m

    # Allocate memory for frequency response samples
    H = np.zeros(N, dtype=complex)

    # Generate channel realizations
    for k in range(N):
        taus = np.random.uniform(tau_min, tau_max, size=num_paths)
        H[k] = np.sum(np.exp(-1j * 2 * np.pi * f * taus))

    # Extract real and imaginary parts
    Re = H.real
    Im = H.imag

    # ---- Plot Real Part ----
    plt.figure(figsize=(6, 4))
    plt.hist(Re, bins=100, density=True, edgecolor="black")
    plt.title("Distribution of Re{H(f)} at 2.5 GHz")
    plt.xlabel("Re{H(f)}")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.savefig("ex1_ReH.jpg")

    # ---- Plot Imaginary Part ----
    plt.figure(figsize=(6, 4))
    plt.hist(Im, bins=100, density=True, edgecolor="black")
    plt.title("Distribution of Im{H(f)} at 2.5 GHz")
    plt.xlabel("Im{H(f)}")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.savefig("ex1_ImH.jpg")

    # ---- Scatter Plot (Constellation) ----
    plt.figure(figsize=(6, 6))
    plt.scatter(Re, Im, s=6, alpha=0.4)
    plt.title("Scatter Plot of H(f) in Complex Plane")
    plt.xlabel("Re{H(f)}")
    plt.ylabel("Im{H(f)}")
    plt.grid(True)
    plt.axis("equal")
    plt.savefig("ex1_ScatterH.jpg")

    print(f"Mean(Re) = {np.mean(Re):.4f}, Std(Re) = {np.std(Re):.4f}")
    print(f"Mean(Im) = {np.mean(Im):.4f}, Std(Im) = {np.std(Im):.4f}")
