import numpy as np

# Create Coordinates
d = 1000
x = np.floor(np.linspace(0, d, d))
y = np.floor(np.linspace(0, d, d))
X, Y = np.meshgrid(x, y)

# Create Forces
A = 10
Cx = 500
Cy = 500
Fx = A * np.exp((X - Cx) ** 2 / (2 * 1000))
Fy = A * np.exp((X - Cy) ** 2 / (2 * 1000))

# Fourier Transform
FFx = np.fft.fft2(Fx)
FFy = np.fft.fft2(Fy)
kx = np.fft.fft2(X)
ky = np.fft.fft2(Y)

# Boussinesq
nu = 0.5
E = 3000
mu = E / (2 * (1 + nu))
k = np.sqrt(kx**2 + ky**2)
k = np.where(k != 0, k, 1e-20)

G = np.zeros((2, 2, d, d), dtype=np.complex128)
G[0][0] = (1 - nu) * kx**2 + ky**2
G[0][1] = -nu * kx * ky
G[1][0] = G[0][1]
G[1][1] = (1 - nu) * ky**2 + kx**2

G = 1 / (mu * k**3)

# Multiply Boussinesq and Forces
