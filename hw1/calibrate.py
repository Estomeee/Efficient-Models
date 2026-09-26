import json
from pathlib import Path

import numpy as np

import torch
import pynvml
from scipy.optimize import least_squares

from hw1.equations import bytes_moved, flops
from hw1.measure_helpers import collect_measurements
from hw1.models import SmallCNN

image_sizes = np.array([32, 48, 96, 160, 288, 416])
batches = np.array([1, 3, 6, 12, 24, 48, 96, 192, 256])


def fit_latency(f, q, measured):
    initial = np.array([
        np.median(f / measured),
        np.median(q / measured),
    ])

    def residual(log_theta):
        compute_rate, bandwidth = np.exp(log_theta)
        predicted = np.maximum(f / compute_rate, q / bandwidth)
        return np.log(predicted) - np.log(measured)

    fits = []
    for scales in [(1, 1), (0.3, 3), (3, 0.3), (0.3, 0.3), (3, 3)]:
        fit = least_squares(
            residual,
            x0=np.log(initial * scales),
            bounds=(np.log(initial) - 20, np.log(initial) + 20),
            max_nfev=2000,
        )
        if fit.success and np.all(np.isfinite(fit.fun)):
            fits.append(fit)

    if not fits:
        raise RuntimeError("Не удалось подобрать коэффициенты latency")

    best = min(fits, key=lambda fit: np.sum(fit.fun**2))
    theta = np.exp(best.x)

    compute_time = f / theta[0]
    memory_time = q / theta[1]

    both_branches = bool(
        np.any(compute_time > memory_time)
        and np.any(memory_time > compute_time)
    )
    if not both_branches:
        print(
            "Предупреждение: все тренировочные точки ограничены "
            "одним ресурсом. Второй коэффициент не определён однозначно."
        )

    return theta, both_branches


def predict_latency(f, q, theta):
    return np.maximum(f / theta[0], q / theta[1])


def fit_energy(measured_energy, predicted_time):
    # Минимизируем сумму квадратов log(P*T) - log(E).
    return float(np.exp(np.mean(
        np.log(measured_energy) - np.log(predicted_time)
    )))


def relative_error(measured, predicted):
    return float(np.mean(np.abs(predicted - measured) / measured))


def get_data(model, sizes, batch_sizes):
    f = np.asarray(flops(sizes, batch_sizes), dtype=float)
    q = np.asarray(bytes_moved(sizes, batch_sizes), dtype=float)

    pynvml.nvmlInit()
    try:
        device = next(model.parameters()).device
        uuid = torch.cuda.get_device_properties(device).uuid

        if isinstance(uuid, bytes):
            uuid = uuid.decode()

        handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

        latency_rows = collect_measurements(
            model, sizes, batch_sizes, mode="latency", log=True)
        energy_rows = collect_measurements(
            model, sizes, batch_sizes, mode='energy', handle=handle, log=True, repeats=1000)

    finally:
        pynvml.nvmlShutdown()

    t = latency_rows[:, 2]
    e = energy_rows[:, 2]

    return f, q, t, e


def main(path):
    model = SmallCNN()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = model.cuda().eval()

    s_grid, b_grid = np.meshgrid(image_sizes, batches, indexing="ij")
    sizes = s_grid.ravel()
    batch_sizes = b_grid.ravel()

    f, q, t, e = get_data(model, sizes, batch_sizes)

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(t))
    n_test = max(1, len(t) // 5)

    test = indices[:n_test]
    train = indices[n_test:]

    theta, both_branches = fit_latency(f[train], q[train], t[train])
    predicted_t = predict_latency(f, q, theta)

    power = fit_energy(e[train], predicted_t[train])
    theta_energy = [float(theta[0]), float(theta[1]), power]
    predicted_e = power * predicted_t

    metrics = {}
    for name, ids in [("train", train), ("test", test)]:
        metrics[name] = {
            "latency_mape": relative_error(t[ids], predicted_t[ids]),
            "energy_mape": relative_error(e[ids], predicted_e[ids]),
        }

    result = {
        "model": "roofline",
        "theta_latency": theta.tolist(),
        "theta_energy": theta_energy,
        "theta_latency_order": [
            "compute_operations_per_second",
            "memory_bytes_per_second",
        ],
        "theta_energy_order": [
            "compute_operations_per_second",
            "memory_bytes_per_second",
            "effective_power_watts",
        ],
        "both_branches_present_in_train": both_branches,
        "train_indices": train.tolist(),
        "test_indices": test.tolist(),
        "metrics": metrics,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)

    print("theta_latency =", tuple(theta))
    print("theta_energy  =", tuple(theta_energy))

    for split, errors in metrics.items():
        print(
            f"{split}: "
            f"latency error = {errors['latency_mape']:.1%}, "
            f"energy error = {errors['energy_mape']:.1%}"
        )

    print("Коэффициенты сохранены в path")


if __name__ == "__main__":
    main('./hw1/results/theta.json')
