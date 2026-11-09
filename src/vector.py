import matplotlib.pyplot as plt
import numpy as np

# Create Coordinates
d = 1000
x = np.floor(np.linspace(0, d, d))
y = np.floor(np.linspace(0, d, d))
X, Y = np.meshgrid(x, y)

# Create Forces
A = 1
Cx = d // 2
Cy = d // 2
Fx = A * np.exp(-((X - Cx) ** 2 + (Y - Cy) ** 2) / (1))
Fy = A * np.exp(-((Y - Cy) ** 2 + (X - Cx) ** 2) / (1))

## Sanity check plot
F = np.sqrt(Fx**2 + Fy**2)
plt.figure()
plt.imshow(F)

# Fourier Transform
FFx = np.fft.fft2(Fx)
FFy = np.fft.fft2(Fy)
kx = np.fft.fftfreq(d, d / d)  # spatial frequency in x-direction
ky = np.fft.fftfreq(d, d / d)  # spatial frequency in y-direction
KX, KY = np.meshgrid(kx, ky)

# Boussinesq
nu = 0.5
E = 9000
mu = E / (2 * (1 + nu))
K = np.sqrt(KX**2 + KY**2)
K = np.where(K != 0, K, 1e-20)

G = np.zeros((2, 2, d, d), dtype=np.complex128)
G[0][0] = (1 - nu) * KX**2 + KY**2
G[0][1] = -nu * KX * KY
G[1][0] = G[0][1]
G[1][1] = (1 - nu) * KY**2 + KX**2

G = G / (mu * K**3)

# Multiply Boussinesq and Forces
Fu = np.zeros((2, d, d), dtype=np.complex128)
Fu[0] = G[0][0] * FFx + G[1][0] * FFy
Fu[1] = G[0][1] * FFx + G[1][1] * FFy

# Transform back to real space
ux = np.fft.ifft2(Fu[0]).real * 10**6
uy = np.fft.ifft2(Fu[1]).real * 10**6

print(np.max(ux), np.max(uy))

## Sanity check plot
u = np.sqrt(ux**2 + uy**2)
plt.figure()
plt.imshow(u)

# plt.show()
