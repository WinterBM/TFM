import random

import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducibility
random.seed(42)

d = 1000
before = np.zeros((d, d))

bead_density = 5  # ~37.5% chance of placing a bead

# Place beads randomly
for row in range(d):
    for col in range(d):
        if random.randrange(1, 9) > bead_density:
            before[row][col] = 1


# Define a rotational force (vortex-like)
def force(x: int, y: int) -> tuple:
    """
    Returns a displacement vector (dx, dy) based on position relative to center.
    This creates a vortex-like rotation.
    """
    # Normalize to avoid huge jumps
    magnitude = 0.9 * (
        1 - np.sqrt(x**2 + y**2) / (d // 2)
    )  # Controls how far particles move
    dx = -x * magnitude
    dy = -y * magnitude
    return int(dx), int(dy)


# Apply force and move beads
after = np.copy(before)

for row in range(d):
    for col in range(d):
        if before[row][col] == 1:
            # Relative position to center
            x = col - d // 2
            y = row - d // 2
            dx, dy = force(x, y)

            new_row = row + dy
            new_col = col + dx

            # Check bounds
            if 0 <= new_row < d and 0 <= new_col < d:
                after[new_row][new_col] += 1
            # If out of bounds, particle vanishes (or you can wrap)

            # Clear original position
            after[row][col] = 0

# Visualization
plt.figure(figsize=(8, 8))
plt.imshow(after, cmap="plasma", interpolation="nearest")
plt.colorbar(label="Number of Particles")
plt.show()
