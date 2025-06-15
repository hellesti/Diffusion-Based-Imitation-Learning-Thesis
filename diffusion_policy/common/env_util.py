import cv2
import numpy as np


def render_env_video(env, states, actions=None):
    observations = states
    imgs = list()
    for i in range(len(observations)):
        state = observations[i]
        env.set_state(state)
        if i == 0:
            env.set_state(state)
        img = env.render()
        # draw action
        if actions is not None:
            action = actions[i]
            coord = (action / 512 * 96).astype(np.int32)
            cv2.drawMarker(img, coord, 
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=8, thickness=1)
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs

def propagate_random_walk(position, velocity, dt, bounds, smoothness=0.1, scale_factors=None, damping_factor=0.0, generator=None, bound_strength=1.0):
    """
    Propagates the current position and velocity of an n-dimensional point one step forward smoothly and randomly within bounds.

    Parameters:
        position (np.ndarray): Current position of the point as a 1D array of shape (n,).
        velocity (np.ndarray): Current velocity of the point as a 1D array of shape (n,).
        bounds (list of tuples): A list of (lower_bound, upper_bound) for each dimension.
        smoothness (float): Determines how smoothly the acceleration varies (smaller = smoother).
        scale_factors (np.ndarray, optional): Scaling factors for acceleration per dimension (1D array of shape (n,)).
        seed (int, optional): Seed for reproducibility.

    Returns:
        tuple: New position and velocity of the point as two 1D arrays.
    """


    bounds = np.array(bounds)
    n = position.shape[0]  # Number of dimensions

    if scale_factors is None:
        scale_factors = np.ones(n)

    # Scale damping
    damping = scale_factors*damping_factor

    # Generate a small random change for acceleration
    if generator is not None:
        acceleration = generator.normal(scale=smoothness, size=n) * scale_factors - damping*velocity
    else:
        acceleration = np.random.normal(scale=smoothness, size=n) * scale_factors - damping*velocity

    # Apply a barrier function to enforce bounds
    for i in range(n):
        if position[i] < bounds[i, 0]:
            acceleration[i] = scale_factors[i] * (bounds[i, 0] - position[i])*bound_strength
        elif position[i] > bounds[i, 1]:
            acceleration[i] = -scale_factors[i] * (position[i] - bounds[i, 1])*bound_strength

    # Update velocity
    velocity = velocity + acceleration*dt

    # Update position
    position = position + velocity*dt

    return position, velocity, acceleration