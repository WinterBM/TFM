import matplotlib.pyplot as plt
import numpy as np
from traction_force.displacement import dft_displacement
from traction_force.plot import plot_dis_vec

import sim
import vector

d = 1000
bead_density = 9
A = 1e-2
E = 9000
nu = 0.49

# Generate before
before = sim.generate_before(d, bead_density)

plt.figure()
plt.title("Before")
plt.imshow(before)

# Generate after
ux, uy = vector.vector_force(d, A, nu, E)
after = sim.generate_after_vector(before, ux, uy)

plt.figure()
plt.title("After")
plt.imshow(after)

# Calculate displacements
result = dft_displacement(before, after, 24, 24)[0]

plot_dis_vec(result, mode="magnitude")

resultx = result[:, :, 0]
resulty = result[:, :, 1]

print(ux)


plt.show()
