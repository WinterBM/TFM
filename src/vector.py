import numpy as np


def traction_patch(r, R, A):
    return np.where(r < R, A, 0)


def vector_force(d=1000, A=200, nu=0.49, E=9000):
    # Create Coordinates (centered at 0 for better physical interpretation)
    x = np.arange(d) - d // 2
    y = np.arange(d) - d // 2
    X, Y = np.meshgrid(x, y)

    # Create Forces
    # Cx = 0
    # Cy = 0
    ## Gaussian
    # Fx = A * np.exp(-((X - Cx) ** 2 + (Y - Cy) ** 2) / (2 * 10**2))
    # Fy = 0 * Fx

    ## Traction patch
    r = np.sqrt(X**2 + Y**2)
    r = np.where(r != 0, r, 1e-10)
    Fx = traction_patch(r, 5, A) * X / r
    Fy = traction_patch(r, 5, A) * Y / r

    # Fourier Transform
    FFx = np.fft.fft2(Fx)
    FFy = np.fft.fft2(Fy)
    kx = np.fft.fftfreq(d, 1) * 2 * np.pi  # spatial frequency in x-direction
    ky = np.fft.fftfreq(d, 1) * 2 * np.pi  # spatial frequency in y-direction
    KX, KY = np.meshgrid(kx, ky)

    # Boussinesq
    mu = E / (2 * (1 + nu))
    K = np.sqrt(KX**2 + KY**2)
    epsilon = 1e-1  # Small parameter for regularization
    K_reg = np.sqrt(K**2 + epsilon**2)

    G = np.zeros((2, 2, d, d), dtype=np.complex128)
    G[0][0] = (1 - nu) * K_reg**2 + nu * KY**2
    G[0][1] = -nu * KX * KY
    G[1][0] = G[0][1]
    G[1][1] = (1 - nu) * K_reg**2 + nu * KX**2

    G = G / (mu * K_reg**3)

    # Multiply Boussinesq and Forces
    Fu = np.zeros((2, d, d), dtype=np.complex128)
    Fu[0] = G[0][0] * FFx + G[1][0] * FFy
    Fu[1] = G[0][1] * FFx + G[1][1] * FFy

    # Transform back to real space
    ux = np.fft.ifft2(Fu[0]).real
    uy = np.fft.ifft2(Fu[1]).real

    # Debug
    print(f"Max Fx: {np.max(Fx):.6f}")
    print(f"Max FFx: {np.max(np.abs(FFx)):.6f}")
    print(f"Max Ux: {np.max(np.abs(ux)):.6f}")

    return ux, uy


def add_gaussian_noise(displacement, noise_level=0.05):
    """
    Add Gaussian noise to displacement field

    Args:
        displacement: 2D array of displacements
        noise_level: Fraction of noise (e.g., 0.05 = 5% of max displacement)

    Returns:
        Noisy displacement field
    """
    max_disp = np.max(np.abs(displacement))
    noise = np.random.normal(0, noise_level * max_disp, displacement.shape)
    return displacement + noise
