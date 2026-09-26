import time
from typing import Literal

import numpy as np
import torch
import pynvml


@torch.inference_mode()
def measure_memory(model, x, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    output = model(x)
    torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)

    del output, x
    return float(peak_bytes)


@torch.inference_mode()
def measure_latency(model, x, device, repeats=50):
    torch.cuda.synchronize(device)
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()

        output = model(x)
        torch.cuda.synchronize(device)

        times.append(time.perf_counter() - start)
        del output

    return float(np.median(times))


@torch.inference_mode()
def measure_energy(model, x, device, handle, repeats=1000):
    torch.cuda.synchronize(device)
    energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

    for _ in range(repeats):
        output = model(x)
        torch.cuda.synchronize(device)
        del output

    energy_after = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

    total_joules = (energy_after - energy_before) / 1000

    if total_joules <= 0:
        raise RuntimeError(
            "Счётчик энергии не изменился. Увеличь repeats."
        )

    return total_joules / repeats


def measure(model, image_size, batch, mode: Literal['latency', 'memory', 'energy'], handle=None, warmup=20, repeats=50):
    if mode == 'energy' and handle is None:
        raise Exception('mode == "energy" and handle is None')

    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(batch, 3, image_size, image_size,
                    device=device, dtype=torch.float32,)

    # Прогрев не входит в измерение
    for _ in range(warmup):
        output = model(x)
        del output  # Не оставляем выход прогрева в памяти.
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    if mode == "latency":
        return measure_latency(model, x, device, repeats=repeats)
    if mode == "memory":
        return measure_memory(model, x, device)
    if mode == "energy":
        return measure_energy(model, x, device, handle, repeats=repeats)


def collect_measurements(model, image_size, batche, mode: Literal['latency', 'memory', 'energy'], handle=None, warmup=20, repeats=50, log=False):
    rows = []

    for s, b in zip(image_size, batche):
        measured = measure(model, s, b, mode=mode, handle=handle, warmup=warmup, repeats=repeats)
        rows.append((s, b, measured))
        if log:
            print(s, b)

    return np.asarray(rows, dtype=float)
