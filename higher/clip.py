""" Not inplace clipping of tensors. """
from functools import partial
import torch
from torch._six import inf


def clip_norm(tensors, max_norm, norm_type=2):
    r"""Clips norm of passed tensors.
    The norm is computed over all tensors together, as if they were
    concatenated into a single vector. Clipped tensors are returned.
    Arguments:
        tensors (Iterable[Tensor]): an iterable of Tensors or a
            single Tensor to be normalized.
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
    Returns:
      Clipped (List[Tensor]) tensors.
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    tensors = list(tensors)
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(t.abs().max() for t in tensors if t is not None)
    else:
        total_norm = 0
        for t in filter(lambda t: t is not None, tensors):
            param_norm = t.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef >= 1:
        return tensors
    return [t.mul(clip_coef) if t is not None else t for t in tensors]


def make_clip_norm_fn(max_norm, norm_type=2):
    r"""Creates clipping function with specified arguments.

    Arguments:
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
    Returns:
      Callable[[List[Tensor]], List[Tensor]] function for clipping tensors.
    """
    return partial(clip_norm, max_norm=max_norm, norm_type=norm_type)
