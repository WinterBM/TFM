import random

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
def force(x: int, y: int, magnitude: float, noise_strength) -> tuple:
    """
    Returns a displacement vector (dx, dy) based on position.
    """

    distance = x**2 + y**2
    if distance == 0:
        return 0, 0
    dx = -y * magnitude / distance
    dy = x * magnitude / distance

    # Add random noise
    noise_x = random.uniform(-noise_strength, noise_strength)
    noise_y = random.uniform(-noise_strength, noise_strength)
    total_dx = dx + noise_x
    total_dy = dy + noise_y

    return (round(total_dx)), (round(total_dy))


# Apply force and move beads
def generate_after(before: np.ndarray, *force_par) -> np.ndarray:
    """Generates particle field after force application.

    Args:
        before: Before Image
        magnitude: Magnitude of force
        force_par: force parameters

    Returns:

    """
    after = np.copy(before)
    d = np.shape(before)[0]
    for row in range(d):
        for col in range(d):
            if before[row][col] == 1:
                # Relative position to center
                x = col - d // 2
                y = row - d // 2
                dx, dy = force(x, y, *force_par)

                new_row = row + dy
                new_col = col + dx

                # Check bounds
                if 0 <= new_row < d and 0 <= new_col < d:
                    after[new_row][new_col] += 1
                # If out of bounds, particle vanishes (or you can wrap)

                # Clear original position
                after[row][col] = 0
    return after
