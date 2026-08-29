import random

import matplotlib.pyplot as plt
import numpy as np


def generate_before(d: int, bead_density: int) -> np.ndarray:
    """Generates a before image.

    Args:
        d: Dimensions (dxd)
        bead_density: A measure for bead density. Has to be between 1 and 10.

    Returns:

    """
    before = np.zeros((d, d))
    for row in range(d):
        for col in range(d):
            if random.randrange(1, 10) > bead_density:
                before[row][col] = 1
    return before


# Define a rotational force (vortex-like)
def force(x: int, y: int, magnitude: float) -> tuple:
    """
    Returns a displacement vector (dx, dy) based on position.
    """

    distance = np.sqrt(x**2 + y**2)
    if distance == 0:
        return 0, 0
    dx = -x * magnitude / distance
    dy = -y * magnitude / distance
    return int(dx), int(dy)


# Apply force and move beads
def generate_after(before, d, magnitude):
    after = np.copy(before)

    for row in range(d):
        for col in range(d):
            if before[row][col] == 1:
                # Relative position to center
                x = col - d // 2
                y = row - d // 2
                dx, dy = force(x, y, magnitude)

                new_row = row + dy
                new_col = col + dx

                # Check bounds
                if 0 <= new_row < d and 0 <= new_col < d:
                    after[new_row][new_col] += 1
                # If out of bounds, particle vanishes (or you can wrap)

                # Clear original position
                after[row][col] = 0
    return after
