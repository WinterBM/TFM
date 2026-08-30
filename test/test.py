import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import sim

steps = 100  # Number of animation steps
fps = 60  # Frames per second
d = 1000
bead_density = 9
strength = 50
noise = 0

# === Setup Animation ===
fig = plt.figure()
im = plt.imshow(np.zeros((d, d)), cmap="gray", interpolation="nearest", vmin=0, vmax=1)

# Global state
current = sim.generate_before(d, bead_density)


# Animation update function
def update(frame):
    global current
    current = sim.generate_after(current, strength, noise)
    im.set_array(current)
    plt.title(f"Simulation Step {frame + 1}")  # No animated=True needed
    return (im,)


# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=steps,
    interval=1000 // fps,
    blit=False,
    repeat_delay=1000,
)

# Show animation
plt.show()
