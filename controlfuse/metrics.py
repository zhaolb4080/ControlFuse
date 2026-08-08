import math
from typing import Dict

import numpy as np
from scipy.ndimage import gaussian_filter, sobel


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float64)
    image = image.astype(np.float64)
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def entropy(image: np.ndarray) -> float:
    values = np.clip(_gray(image) * 255.0, 0, 255).astype(np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probability = histogram / histogram.sum()
    probability = probability[probability > 0]
    return float(-(probability * np.log2(probability)).sum())


def standard_deviation(image: np.ndarray) -> float:
    return float(_gray(image).std() * 255.0)


def spatial_frequency(image: np.ndarray) -> float:
    value = _gray(image) * 255.0
    rf = np.sqrt(np.mean(np.diff(value, axis=0) ** 2))
    cf = np.sqrt(np.mean(np.diff(value, axis=1) ** 2))
    return float(np.sqrt(rf * rf + cf * cf))


def average_gradient(image: np.ndarray) -> float:
    value = _gray(image) * 255.0
    gx = np.diff(value, axis=1)[:-1]
    gy = np.diff(value, axis=0)[:, :-1]
    return float(np.sqrt((gx * gx + gy * gy) * 0.5).mean())


def _vifp(reference: np.ndarray, distorted: np.ndarray) -> float:
    ref = _gray(reference) * 255.0
    dist = _gray(distorted) * 255.0
    sigma_nsq = 2.0
    numerator = 0.0
    denominator = 0.0
    for scale in range(1, 5):
        n = 2 ** (5 - scale) + 1
        sigma = n / 5.0
        if scale > 1:
            ref = gaussian_filter(ref, sigma)[::2, ::2]
            dist = gaussian_filter(dist, sigma)[::2, ::2]
        mu1 = gaussian_filter(ref, sigma)
        mu2 = gaussian_filter(dist, sigma)
        sigma1_sq = gaussian_filter(ref * ref, sigma) - mu1 * mu1
        sigma2_sq = gaussian_filter(dist * dist, sigma) - mu2 * mu2
        sigma12 = gaussian_filter(ref * dist, sigma) - mu1 * mu2
        sigma1_sq = np.maximum(sigma1_sq, 0)
        sigma2_sq = np.maximum(sigma2_sq, 0)
        gain = sigma12 / (sigma1_sq + 1e-10)
        noise = sigma2_sq - gain * sigma12
        gain[sigma1_sq < 1e-10] = 0
        noise[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        gain[sigma2_sq < 1e-10] = 0
        noise[gain < 0] = sigma2_sq[gain < 0]
        gain = np.maximum(gain, 0)
        noise = np.maximum(noise, 1e-10)
        numerator += np.log1p(gain * gain * sigma1_sq / (noise + sigma_nsq)).sum()
        denominator += np.log1p(sigma1_sq / sigma_nsq).sum()
    return float(numerator / (denominator + 1e-10))


def vif_fusion(infrared: np.ndarray, visible: np.ndarray, fused: np.ndarray) -> float:
    return _vifp(infrared, fused) + _vifp(visible, fused)


def _edge_strength_angle(image: np.ndarray):
    value = _gray(image)
    gx = sobel(value, axis=1, mode="reflect")
    gy = sobel(value, axis=0, mode="reflect")
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def _edge_preservation(source: np.ndarray, fused: np.ndarray):
    gs, angles = _edge_strength_angle(source)
    gf, anglef = _edge_strength_angle(fused)
    ratio = np.minimum(gs, gf) / (np.maximum(gs, gf) + 1e-10)
    difference = np.abs((angles - anglef + math.pi / 2) % math.pi - math.pi / 2)
    angle = 1 - difference / (math.pi / 2)
    angle = np.clip(angle, 0, 1)
    qg = 0.9994 / (1 + np.exp(-15 * (ratio - 0.5)))
    qa = 0.9879 / (1 + np.exp(-22 * (angle - 0.8)))
    return qg * qa, gs


def qabf(infrared: np.ndarray, visible: np.ndarray, fused: np.ndarray) -> float:
    qa, wa = _edge_preservation(infrared, fused)
    qb, wb = _edge_preservation(visible, fused)
    return float((qa * wa + qb * wb).sum() / (wa + wb).sum().clip(min=1e-10))


def all_metrics(infrared: np.ndarray, visible: np.ndarray, fused: np.ndarray) -> Dict[str, float]:
    return {
        "EN": entropy(fused),
        "SD": standard_deviation(fused),
        "SF": spatial_frequency(fused),
        "AG": average_gradient(fused),
        "VIF": vif_fusion(infrared, visible, fused),
        "Qabf": qabf(infrared, visible, fused),
    }
