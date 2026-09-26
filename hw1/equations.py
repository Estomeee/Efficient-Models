import numpy as np


def flops(image_size, batch):
    s = np.asarray(image_size, dtype=float)
    b = np.asarray(batch, dtype=float)
    return b * (17_730 * s**2 + 313_600)


def memory(image_size, batch):
    s = np.asarray(image_size, dtype=float)
    b = np.asarray(batch, dtype=float)
    return 4 * (1_040_324 + b * (26 * s**2 + 868))


def bytes_moved(image_size, batch):
    s = np.asarray(image_size, dtype=float)
    b = np.asarray(batch, dtype=float)
    return 4 * (1_040_324 + b * (91 * s**2 + 2_148))


def latency(image_size, batch, theta):
    # theta: (operations per second, bytes per second)
    comp = flops(image_size, batch)
    mem = bytes_moved(image_size, batch)

    return np.maximum(comp / theta[0], mem / theta[1])


def energy(image_size, batch, theta_energy):
    # theta_energy: (operations per second, bytes per second, watts)
    lat = latency(image_size, batch, theta_energy[:2])
    return theta_energy[2] * lat