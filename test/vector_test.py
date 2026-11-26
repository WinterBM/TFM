import matplotlib.pyplot as plt
import numpy as np
from traction_force.displacement import dft_displacement
from traction_force.force import tract
from traction_force.plot import plot_dis_vec

import sim
import vector

d = 500
bead_density = 9
A = 10000
E = 1200
nu = 0.49
noise = 5
background = 100
sigma = 1

# Generate before
before, pos = sim.generate_before(
    d, bead_density, noise_level=noise, background=background, sigma=sigma
)

plt.figure()
plt.title("Before")
plt.imshow(before, cmap="gray")

# Generate after
ux, uy = vector.vector_force(d, A, nu, E)
# ux = vector.add_gaussian_noise(ux, noise)
# uy = vector.add_gaussian_noise(uy, noise)
after = sim.generate_after_vector(before, pos, ux, uy, background, d, noise, A, sigma)

plt.figure()
plt.title("After")
plt.imshow(after, cmap="gray")

# Calculate displacements
result = dft_displacement(before, after, 12, 12)[0]


plot_dis_vec(result, mode="magnitude")

# Calculate tractions
tractions = tract(result, E, nu, 1, 1e-15, 70, 70, 2)

t = tractions["tractions"].real

plot_dis_vec(t, mode="magnitude")

# print(np.max(result))


plt.show()
