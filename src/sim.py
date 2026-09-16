import random
import math

import numpy as np
from scipy.special import ellipe, ellipk


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
    # Rotation
    dx = -y
    dy = x
    # Inwards force
    dx -= x
    dy -= y
    # Scaling
    dx *= magnitude / distance
    dy *= magnitude / distance

    # Add random noise
    noise_x = random.uniform(-noise_strength, noise_strength)
    noise_y = random.uniform(-noise_strength, noise_strength)
    total_dx = dx + noise_x
    total_dy = dy + noise_y

    return total_dx, total_dy


def Heavy_Side(x):
    if x < 0:
        return 0
    else:
        return 1


def traction_patch(x, y, R=50.0, nu=0.5, f0=1, E=1):
    r = np.sqrt(x**2 + y**2)
    epsilon = r**2 / R**2
    theta = np.arcsin(y / r)
    if r < 1e-6:
        return 0, 0
    if r < R:
        N1 = 4 * ellipk(epsilon)
        N2 = (
            4
            * np.cos(2 * theta)
            * ((r**2 + R**2) * ellipk(epsilon) + (r**2 - R**2) * ellipe(epsilon))
        ) / (3 * r**2)
        N2 += 4 * np.sin(theta) ** 2 * ellipk(epsilon)
        N3 = (
            2
            * np.cos(2 * theta)
            * (
                (r**2 + 2 * R**2) * ellipk(epsilon)
                + 2 * (r**2 - R**2) * ellipe(epsilon)
            )
        ) / (3 * r**2)
        N4 = (
            -N2
            - 4 * np.sin(theta) ** 2 * ellipk(epsilon)
            + 4 * np.cos(theta) ** 2 * ellipk(epsilon)
        )
    else:
        epsilon2 = R**2 / r**2
        N1 = (4 * (r**2 * ellipk(epsilon2)) + (R**2 - r**2) * ellipe(epsilon2)) / (
            r * R
        )
        N2 = (
            (6 * r**2 - 2 * (r**2 - 2 * R**2) * np.cos(theta)) * ellipk(epsilon2)
            + 2 * (r**2 - R**2) * (np.cos(2 * theta) - 3) * ellipe(epsilon2)
        ) / (3 * r * R)
        N3 = (
            2
            * np.sin(2 * theta)
            * ((r**2 - 2 * R**2) * ellipk(epsilon2) + (R**2 - r**2) * ellipe(epsilon2))
        ) / (3 * r * R)
        N4 = (
            (6 * r**2 - 2 * (r**2 - 2 * R**2) * np.cos(theta)) * ellipk(epsilon2)
            - 2 * (r**2 - R**2) * (np.cos(2 * theta) - 3) * ellipe(epsilon2)
        ) / (3 * r * R)

    dx = ((R * (1 + nu)) / (np.pi * E)) * (
        ((1 - nu) * N1 + nu * N2) * f0 * np.cos(theta) - N3 * f0 * np.sin(theta)
    )
    dy = ((R * (1 + nu)) / (np.pi * E)) * (
        -nu * N3 * f0 * np.cos(theta) + ((1 - nu) * N1 + nu * N4) * f0 * np.sin(theta)
    )
    print(dx, dy)
    if math.isnan(dx) or math.isnan(dy) or math.isinf(dx) or math.isinf(dy):
        return 0, 0
    return dx, dy


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
            if before[row][col] > 0:
                # Relative position to center
                x = col - d // 2
                y = row - d // 2
                dx, dy = traction_patch(x, y, f0=0.1)

                step_x = int(np.rint(dx))
                step_y = int(np.rint(dy))

                new_row = row + step_y
                new_col = col + step_x

                if 0 <= new_row < d and 0 <= new_col < d:
                    after[new_row][new_col] += 1

                # Clear original position
                after[row][col] -= 1
    return after
