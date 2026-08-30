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
            if random.randrange(1, 11) > bead_density:
                before[row][col] = 1
    return before


def force(
    x: int,
    y: int,
    magnitude: float,
    noise_strength: float,
):
    """
    Returns a displacement vector (dx, dy) based on position.
    """

    distance = np.sqrt(x**2 + y**2)
    if distance < 1e-6:
        return 0, 0
    dx = -x * magnitude / distance
    dy = -y * magnitude / distance

    # Add random noise
    noise_x = random.uniform(-noise_strength, noise_strength)
    noise_y = random.uniform(-noise_strength, noise_strength)
    total_dx = dx + noise_x
    total_dy = dy + noise_y

    return total_dx, total_dy


# Apply force and move beads
def generate_after(
    before: np.ndarray, magnitude: float, noise_strength: float
) -> np.ndarray:
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
                x = col - d / 2
                y = row - d / 2
                dx, dy = force(x, y, magnitude, noise_strength)

                # Add small random walk toward center
                step_x = int(np.rint(dx))
                step_y = int(np.rint(dy))

                # or use probabilistic step
                if random.random() < 0.5:
                    step_x = 0
                if random.random() < 0.5:
                    step_y = 0

                new_row = row + step_y
                new_col = col + step_x

                if (
                    0 <= new_row < d
                    and 0 <= new_col < d
                    and after[new_row][new_col] == 0
                ):
                    after[new_row][new_col] = 1
                    after[row][col] = 0
                elif (
                    0 <= new_row < d
                    and 0 <= new_col < d
                    and after[new_row][new_col] == 1
                ):
                    print("Hi")

                # Clear original position
                after[row][col] = 0
    return after
