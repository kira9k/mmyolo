import numpy as np
import torch
import cv2

import random
from mmyolo.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness



@TRANSFORMS.register_module()
class ThermalJitter(BaseTransform):
    """Bias + Gain + Gamma augmentation for thermal images.

    Simulates temperature drift (bias), contrast variation (gain),
    and nonlinear sensor response (gamma). Operates on all 3 channels
    simultaneously since thermal pseudo-RGB channels are identical.

    Args:
        bias_range (tuple): Min/max for additive bias, applied as fraction
            of 255. E.g. (-0.1, 0.1) means bias in [-25.5, 25.5] for uint8.
        gain_range (tuple): Min/max multiplicative gain factor.
        gamma_prob (float): Probability of applying gamma correction.
        gamma_range (tuple): Min/max gamma exponent.
    """

    def __init__(self,
                 bias_range=(-0.1, 0.1),
                 gain_range=(0.8, 1.3),
                 gamma_prob=0.6,
                 gamma_range=(0.7, 1.4)):
        self.bias_range = bias_range
        self.gain_range = gain_range
        self.gamma_prob = gamma_prob
        self.gamma_range = gamma_range

    @cache_randomness
    def _random_params(self):
        bias = random.uniform(*self.bias_range) * 255.0
        gain = random.uniform(*self.gain_range)
        apply_gamma = random.random() < self.gamma_prob
        gamma = random.uniform(*self.gamma_range) if apply_gamma else 1.0
        return bias, gain, apply_gamma, gamma

    def transform(self, results):
        img = results['img'].astype(np.float32)
        bias, gain, apply_gamma, gamma = self._random_params()

        # Bias (temperature shift)
        img = img + bias

        # Gain (contrast) — center around mean
        mean = img.mean()
        img = (img - mean) * gain + mean

        # Gamma correction (nonlinear contrast)
        if apply_gamma:
            img = np.clip(img, 0, 255)
            img = np.power(img / 255.0, gamma) * 255.0

        results['img'] = np.clip(img, 0, 255).astype(np.uint8)
        return results


@TRANSFORMS.register_module()
class ThermalSensorNoise(BaseTransform):
    """Sensor noise simulation for thermal images.

    Adds Gaussian noise (sensor readout noise) and salt & pepper noise
    (dead/hot pixels). Applied identically to all channels.

    Args:
        gaussian_prob (float): Probability of adding Gaussian noise.
        sigma_range (tuple): Min/max sigma for Gaussian noise (in 0-255 scale).
        salt_pepper_prob (float): Probability of adding salt & pepper noise.
        salt_pepper_ratio (float): Fraction of pixels affected by S&P noise.
    """

    def __init__(self,
                 gaussian_prob=0.7,
                 sigma_range=(0.8, 7.0),
                 salt_pepper_prob=0.3,
                 salt_pepper_ratio=0.002):
        self.gaussian_prob = gaussian_prob
        self.sigma_range = sigma_range
        self.salt_pepper_prob = salt_pepper_prob
        self.salt_pepper_ratio = salt_pepper_ratio

    @cache_randomness
    def _random_params(self):
        apply_gauss = random.random() < self.gaussian_prob
        sigma = random.uniform(*self.sigma_range) if apply_gauss else 0.0
        apply_sp = random.random() < self.salt_pepper_prob
        return apply_gauss, sigma, apply_sp

    def transform(self, results):
        img = results['img'].astype(np.float32)

        apply_gauss, sigma, apply_sp = self._random_params()

        # Gaussian noise
        if apply_gauss:
            noise = np.random.randn(*img.shape).astype(np.float32) * sigma
            img = img + noise

        # Salt & Pepper (dead/hot pixels)
        if apply_sp:
            mask = np.random.rand(*img.shape[:2]) < self.salt_pepper_ratio
            # Same mask for all channels to simulate real dead/hot pixels
            mask_3ch = np.stack([mask] * img.shape[2], axis=-1) if img.ndim == 3 else mask
            random_vals = np.random.randint(0, 256, size=img.shape).astype(np.float32)
            img = np.where(mask_3ch, random_vals, img)

        results['img'] = np.clip(img, 0, 255).astype(np.uint8)
        return results


@TRANSFORMS.register_module()
class AtmosphericEffect(BaseTransform):
    """Simulates atmospheric effects (haze, distance degradation) on thermal.

    Models the effect where distant objects appear more uniform due to
    atmospheric attenuation. Applied uniformly across all channels.

    Args:
        prob (float): Probability of applying the effect.
        alpha_range (tuple): Min/max for atmospheric alpha blending factor.
            Higher alpha = stronger haze effect.
    """

    def __init__(self, prob=0.5, alpha_range=(0.05, 0.22)):
        self.prob = prob
        self.alpha_range = alpha_range

    @cache_randomness
    def _random_params(self):
        apply = random.random() < self.prob
        alpha = random.uniform(*self.alpha_range) if apply else 0.0
        return apply, alpha

    def transform(self, results):
        apply, alpha = self._random_params()
        if apply:
            img = results['img'].astype(np.float32)
            mean_val = img.mean()
            img = img * (1 - alpha) + mean_val * alpha
            results['img'] = np.clip(img, 0, 255).astype(np.uint8)
        return results


@TRANSFORMS.register_module()
class LocalContrast(BaseTransform):
    """Local contrast enhancement via unsharp masking (CLAHE-like).

    Sharpens local details by amplifying high-frequency components.
    Applied uniformly across all channels.

    Args:
        prob (float): Probability of applying local contrast enhancement.
        strength (float): Unsharp mask strength factor.
        kernel_size (int): Gaussian kernel size for blur.
        sigma (float): Gaussian sigma for blur.
    """

    def __init__(self, prob=0.6, strength=0.8, kernel_size=5, sigma=2.5):
        self.prob = prob
        self.strength = strength
        self.kernel_size = kernel_size
        self.sigma = sigma

    @cache_randomness
    def _random_apply(self):
        return random.random() < self.prob

    def transform(self, results):
        if self._random_apply():
            img = results['img'].astype(np.float32)
            blurred = cv2.GaussianBlur(
                img, (self.kernel_size, self.kernel_size), self.sigma)
            img = img + self.strength * (img - blurred)
            results['img'] = np.clip(img, 0, 255).astype(np.uint8)
        return results


@TRANSFORMS.register_module()
class ThermalGaussianBlur(BaseTransform):
    """Gaussian blur augmentation for thermal images.

    Simulates defocus and motion blur artifacts common in thermal sensors.

    Args:
        prob (float): Probability of applying blur.
        sigma_range (tuple): Min/max sigma for Gaussian kernel.
        kernel_size (int): Kernel size for Gaussian blur (must be odd).
    """

    def __init__(self, prob=0.4, sigma_range=(0.5, 2.5), kernel_size=3):
        self.prob = prob
        self.sigma_range = sigma_range
        self.kernel_size = kernel_size

    @cache_randomness
    def _random_params(self):
        apply = random.random() < self.prob
        sigma = random.uniform(*self.sigma_range) if apply else 0.0
        return apply, sigma

    def transform(self, results):
        apply, sigma = self._random_params()
        if apply:
            results['img'] = cv2.GaussianBlur(
                results['img'],
                (self.kernel_size, self.kernel_size),
                sigma)
        return results


@TRANSFORMS.register_module()
class ThermalRandomErasing(BaseTransform):
    """Random erasing for thermal images.

    Masks rectangular regions with a value typical for thermal sensors
    (random value between local min and max). Applied identically to all
    channels to simulate sensor dead zones or occlusions.

    Args:
        prob (float): Probability of applying random erasing.
        num_area_range (tuple): Min/max number of erased rectangles.
        hole_size_range (tuple): Min/max size (pixels) of each rectangle.
        min_size (int): Minimum image dimension for valid erasing.
    """

    def __init__(self,
                 prob=0.4,
                 num_area_range=(1, 3),
                 hole_size_range=(24, 80),
                 min_size=32):
        self.prob = prob
        self.num_area_range = num_area_range
        self.hole_size_range = hole_size_range
        self.min_size = min_size

    @cache_randomness
    def _random_params(self):
        apply = random.random() < self.prob
        num_areas = random.randint(*self.num_area_range) if apply else 0
        return apply, num_areas

    def transform(self, results):
        apply, num_areas = self._random_params()
        if not apply:
            return results

        img = results['img'].astype(np.float32)
        h, w = img.shape[:2]

        for _ in range(num_areas):
            x = random.randint(0, max(w - self.min_size, 1))
            y = random.randint(0, max(h - self.min_size, 1))
            bh = random.randint(*self.hole_size_range)
            bw = random.randint(*self.hole_size_range)
            bh = min(bh, h - y)
            bw = min(bw, w - x)

            # Fill with random value in local intensity range
            region = img[y:y + bh, x:x + bw]
            fill_val = random.uniform(region.min(), region.max())
            img[y:y + bh, x:x + bw] = fill_val

        results['img'] = img.astype(np.uint8)
        return results

@TRANSFORMS.register_module()
class MyInvert(BaseTransform):
    def __init__(self, p=0.5):
        self.prob = p

    def transform(self, results: dict) -> dict:
        if np.random.random() > self.prob:
            return results
            
        img = results['img']
        inverted_img = 255 - img
        
        results['img'] = inverted_img
        return results
    

# @TRANSFORMS.register_module()
# class FocusTransform:
#     """Apply Focus-style preprocessing via pixel_unshuffle.
    
#     [3, H, W] -> [12, H/2, W/2]
#     """
#     def __call__(self, results: dict) -> dict:
#         img = results['img']  # numpy [H, W, 3]
        
#         # numpy -> tensor [1, 3, H, W]
#         img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        
#         # pixel_unshuffle: [1, 3, H, W] -> [1, 12, H/2, W/2]
#         patch = torch.nn.functional.pixel_unshuffle(img_tensor, downscale_factor=4)
        
#         # tensor -> numpy [H/2, W/2, 12]
#         results['img'] = patch.squeeze(0).permute(1, 2, 0).numpy()
#         results['img_shape'] = results['img'].shape[:2]
        
#         return results