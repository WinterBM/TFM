import numpy as np


def vector_force(d=1000, A=1e-2, nu=0.49, E=9000):
    # Create Coordinates
    x = np.floor(np.linspace(0, d, d))
    y = np.floor(np.linspace(0, d, d))
    X, Y = np.meshgrid(x, y)

    # Create Forces
    Cx = d // 2
    Cy = d // 2
    Fx = A * np.exp(-((X - Cx) ** 2 + (Y - Cy) ** 2) / (1))
    Fy = A * np.exp(-((Y - Cy) ** 2 + (X - Cx) ** 2) / (1))

    # Fourier Transform
    FFx = np.fft.fft2(Fx)
    FFy = np.fft.fft2(Fy)
    kx = np.fft.fftfreq(d, d / d)  # spatial frequency in x-direction
    ky = np.fft.fftfreq(d, d / d)  # spatial frequency in y-direction
    KX, KY = np.meshgrid(kx, ky)

    # Boussinesq
    mu = E / (2 * (1 + nu))
    K = np.sqrt(KX**2 + KY**2)
    K = np.where(K != 0, K, 1e-20)

    G = np.zeros((2, 2, d, d), dtype=np.complex128)
    G[0][0] = (1 - nu) * K**2 + nu * KY**2
    G[0][1] = -nu * KX * KY
    G[1][0] = G[0][1]
    G[1][1] = (1 - nu) * K**2 + nu * KX**2

    G = G / (mu * K)

    # Multiply Boussinesq and Forces
    Fu = np.zeros((2, d, d), dtype=np.complex128)
    Fu[0] = G[0][0] * FFx + G[1][0] * FFy
    Fu[1] = G[0][1] * FFx + G[1][1] * FFy

    # Transform back to real space
    ux = np.fft.ifft2(Fu[0]).real
    uy = np.fft.ifft2(Fu[1]).real

    return ux, uy
