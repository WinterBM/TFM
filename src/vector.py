import numpy as np


def traction_patch(r, R, A):
    return np.where(r < R, A, 0)


def vector_force(d=1000, A=1e-2, nu=0.49, E=9000):
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
    kx = np.fft.fftfreq(d, d / d) * 2 * np.pi  # spatial frequency in x-direction
    ky = np.fft.fftfreq(d, d / d) * 2 * np.pi  # spatial frequency in y-direction
    KX, KY = np.meshgrid(kx, ky)

    # Boussinesq
    mu = E / (2 * (1 + nu))
    K = np.sqrt(KX**2 + KY**2)
    K = np.where(K != 0, K, 1e-10)

    G = np.zeros((2, 2, d, d), dtype=np.complex128)
    G[0][0] = (1 - nu) * K**2 + nu * KY**2
    G[0][1] = -nu * KX * KY
    G[1][0] = G[0][1]
    G[1][1] = (1 - nu) * K**2 + nu * KX**2

    G = G / (mu * K**3)

    # Multiply Boussinesq and Forces
    Fu = np.zeros((2, d, d), dtype=np.complex128)
    Fu[0] = G[0][0] * FFx + G[1][0] * FFy
    Fu[1] = G[0][1] * FFx + G[1][1] * FFy

    # Transform back to real space
    ux = np.fft.ifft2(Fu[0]).real
    uy = np.fft.ifft2(Fu[1]).real

    return ux, uy
