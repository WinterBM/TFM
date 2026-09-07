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


def force_cell(
    x: float,
    y: float,
    cell_center_x: float,
    cell_center_y: float,
    magnitude: float,
    radius: float,
    noise_strength: float = 0.0,
):
    """
    Simulate a cell pulling on the substrate with radial inward force.

    Args:
        x, y: Position relative to substrate center
        cell_center_x, cell_center_y: Center of the cell (in substrate coordinates)
        magnitude: Peak force magnitude (at cell edge)
        radius: Radius of the cell (in pixels)
        noise_strength: Add random noise to displacement (for realism)

    Returns:
        (dx, dy): Displacement vector (in pixels)
    """
    # Relative position to cell center
    dx_cell = x - cell_center_x
    dy_cell = y - cell_center_y
    r = np.sqrt(dx_cell**2 + dy_cell**2)

    # Force only inside cell
    if r > radius:
        return 0.0, 0.0

    # Radial inward force (Hertz-like, but simplified)
    # Force decreases from edge to center
    force_factor = magnitude * (1 - r / radius)  # Linear decay from edge to center

    # Direction: inward
    if r < 1e-6:
        return 0.0, 0.0

    # Normalize direction
    dir_x = -dx_cell / r
    dir_y = -dy_cell / r

    # Apply force
    disp_x = force_factor * dir_x
    disp_y = force_factor * dir_y

    # Add noise (realistic for imaging)
    disp_x += random.uniform(-noise_strength, noise_strength)
    disp_y += random.uniform(-noise_strength, noise_strength)

    return disp_x, disp_y


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
                dx, dy = force(x, y, magnitude, noise_strength)

                step_x = int(np.rint(dx))
                step_y = int(np.rint(dy))

                new_row = row + step_y
                new_col = col + step_x

                if 0 <= new_row < d and 0 <= new_col < d:
                    after[new_row][new_col] += 1

                # Clear original position
                after[row][col] -= 1
    return after
