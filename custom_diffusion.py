import torch
import numpy as np
import math
from numba import njit, jit, cuda

_alpha_bar_cache = {}

def _beta(current_t: float, max_t: float, beta_small: float, beta_large: float):
    return beta_small + (current_t / max_t) * (beta_large - beta_small)

def _alpha(current_t: float, max_t: float, beta_small: float, beta_large: float):
    return 1.0 - _beta(current_t, max_t, beta_small, beta_large)

def _alpha_bar_create_cache(max_t: float, beta_small: float, beta_large: float):
    max_t_int = int(max_t)
    result = []
    for i in range(max_t_int):
        result.append(math.prod([_alpha(float(i), max_t, beta_small, beta_large) for j in range(i)]))
    return result

def _alpha_bar(current_t: float, max_t: float, beta_small: float, beta_large: float):
    cache_key = f"{max_t}_{beta_small}_{beta_large}"
    if not (cache_key in _alpha_bar_cache):
        _alpha_bar_cache[cache_key] = _alpha_bar_create_cache(max_t, beta_small, beta_large)
    current_t_int = int(current_t)
    return _alpha_bar_cache[cache_key][current_t_int]

def create_random_time_steps(number_of_items: int, max_t: int, device: torch.device):
    #return torch.from_numpy(np.random.choice(max_t, size=(number_of_items,), p=np.ones([max_t])/max_t)).to(device)
    return torch.randint(low=0, high=max_t, size=(number_of_items,)).to(device)

def diffuse_images(images: torch.Tensor, time_steps: torch.Tensor, max_t: int, beta_small: float, beta_large: float):
    noised_images = torch.empty_like(images)
    source_noise = torch.randn_like(images)
    for i in range(images.shape[0]):
        current_alpha_bar = _alpha_bar(float(time_steps[i]), float(max_t), beta_small, beta_large)
        noised_images[i] = (math.sqrt(current_alpha_bar) * images[i]) + (math.sqrt(1.0 - current_alpha_bar) * source_noise[i])

    return noised_images, source_noise

def estimate_noise(model, noised_images, time_steps, z_sem=None):
    result = None
    if (z_sem is not None):
        result = model.forward(x=noised_images, t=time_steps, cond=z_sem)
    else:
        result = model.forward(x=noised_images, t=time_steps)
    if (isinstance(result, torch.Tensor)):
        return result
    else:
        return result.pred

def estimate_noise_paper(model, noised_images, time_steps, z_sem=None):
    if (z_sem is not None):
        return model.forward(x=noised_images, t=time_steps, cond=z_sem).pred
    else:
        return model.forward(x=noised_images, t=time_steps).pred

def encode_semantic(semantic_encoder, images):
    return semantic_encoder.forward(images)

# We return only the mean of this distribution - its variance is 0
# We also expect an input dimension for the images of (3,size,size) - no batch dimension
def q_distribution_single_image(x_t: torch.Tensor, x_0: torch.Tensor, current_t: float, max_t: float, beta_small: float, beta_large: float):
    _alpha_bar_t = _alpha_bar(current_t, max_t, beta_small, beta_large)
    _sqrt_alpha_bar_t = math.sqrt(_alpha_bar_t)
    _sqrt_one_minus_alpha_bar_t = math.sqrt(1.0 - _alpha_bar_t)

    _alpha_bar_t_minus_1 = _alpha_bar(current_t - 1, max_t, beta_small, beta_large)
    _sqrt_alpha_bar_t_minus_1 = math.sqrt(_alpha_bar_t_minus_1)
    _sqrt_one_minus_alpha_bar_t_minus_1 = math.sqrt(1.0 - _alpha_bar_t_minus_1)

    last_quarter_of_equation = (x_t - (_sqrt_alpha_bar_t * x_0)) / _sqrt_one_minus_alpha_bar_t

    return ((_sqrt_alpha_bar_t_minus_1 * x_0) + (_sqrt_one_minus_alpha_bar_t_minus_1 * last_quarter_of_equation))

#Like q_distribution_single_image, but x_t, x_0 have a batch dimension, and current_t is a 1-d tensor
def q_distribution_multiple_images(x_t: torch.Tensor, x_0: torch.Tensor, current_t: torch.Tensor, max_t: int, beta_small: float, beta_large: float):
    # We can use either x_t or x_0 here, as long as they are on the same device
    result = torch.empty_like(x_0)
    for i in range(current_t.shape[0]):
        result[i] = q_distribution_single_image(x_t[i], x_0[i], float(current_t[i]), float(max_t), beta_small, beta_large)
    return result

# We expect this to be batched like in q_distribution_multiple_images
def f_theta_multiple_images(ddim_model, x_t, t, z_sem, max_t, beta_small, beta_large):
    # We can use either x_t or x_0 here, as long as they are on the same device
    result = torch.empty_like(x_t)
    e_theta = estimate_noise(ddim_model, x_t, t, z_sem)

    for i in range(t.shape[0]):
        _alpha_bar_t = _alpha_bar(int(t[i]), max_t, beta_small, beta_large)
        _sqrt_alpha_bar_t = math.sqrt(_alpha_bar_t)
        _sqrt_one_minus_alpha_bar_t = math.sqrt(1.0 - _alpha_bar_t)
        current_result = (1.0 / _sqrt_alpha_bar_t) * (x_t[i] - (_sqrt_one_minus_alpha_bar_t * e_theta[i]))
        result[i] = current_result
    return result

#You can use this for one image by unsqueezing/creating the batch dimension for x_t and z_sem
def denoise_process_multiple_images(ddim_model, x_t, z_sem, max_t, beta_small, beta_large):
    batch_size = x_t.shape[0]
    time_steps = torch.arange(start=max_t - 1, end=-1, step=-1, device=x_t.device)
    for i in range(time_steps.shape[0]):
        current_t_repeat = time_steps[i].repeat(batch_size)
        current_t = int(time_steps[i])
        assert current_t >= 0, "What the ****"
        f_theta_result = f_theta_multiple_images(ddim_model, x_t, current_t_repeat, z_sem, max_t, beta_small, beta_large)
        if current_t > 0:
            x_t = q_distribution_multiple_images(x_t, f_theta_result, current_t_repeat, max_t, beta_small, beta_large)
        elif current_t == 0:
            x_t = f_theta_result
    return x_t

#You can use this for one image by unsqueezing/creating the batch dimension for x_t and z_sem
def stochastic_encode_process_multiple_images(ddim_model, x_0, z_sem, max_t, beta_small, beta_large):
    batch_size = x_0.shape[0]
    time_steps = torch.arange(start=0, end=max_t, step=1, device=x_0.device)
    for i in range(time_steps.shape[0]):
        current_t_repeat = time_steps[i].repeat(batch_size)
        current_t = int(time_steps[i])
        _alpha_bar_t_plus_1 = _alpha_bar(current_t + 1, max_t, beta_small, beta_large)
        _sqrt_alpha_bar_t_plus_1 = math.sqrt(_alpha_bar_t_plus_1)
        _sqrt_one_minus_alpha_bar_t_plus_1 = math.sqrt(1.0 - _alpha_bar_t_plus_1)

        f_theta_result = f_theta_multiple_images(ddim_model, x_0, current_t_repeat, z_sem, max_t, beta_small, beta_large)
        e_theta = estimate_noise(ddim_model, x_0, current_t_repeat, z_sem)
        x_0 = (_sqrt_alpha_bar_t_plus_1 * f_theta_result) + (_sqrt_one_minus_alpha_bar_t_plus_1 * e_theta)
    return x_0
