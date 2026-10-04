"""Pure-PyTorch stand-in for the parts of ``torch_scatter`` that DPVO uses.

Why: PyG stopped publishing ``torch_scatter`` wheels after torch 2.9; for
torch 2.14.1+cu130 only ``pyg_lib`` exists. DPVO imports ``scatter_sum``,
``scatter_softmax`` and ``scatter_max``. Upstream torch_scatter 2.1.2 already
implements ``scatter_sum`` and ``scatter_softmax`` in Python (on top of
``Tensor.scatter_add_`` and ``scatter_max``). Only ``scatter_max/min`` call
compiled ops, and here they use ``Tensor.scatter_reduce_``.

Semantics follow torch_scatter 2.1.2:
  * ``index`` is broadcast against ``src`` exactly like ``torch_scatter.utils.broadcast``.
  * Output size along ``dim`` is ``dim_size``, or ``index.max() + 1``.
  * Empty groups give 0 for sum, mean, max and min. Their arg index is ``src.size(dim)``.
"""
from typing import Optional, Tuple

import torch

__version__ = "2.1.2+shim"

__all__ = [
    "scatter", "scatter_sum", "scatter_add", "scatter_mul", "scatter_mean",
    "scatter_max", "scatter_min", "scatter_softmax", "scatter_log_softmax",
]


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def _out_size(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: Optional[int]):
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    return size


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        out = torch.zeros(_out_size(src, index, dim, dim_size), dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


def scatter_add(src, index, dim=-1, out=None, dim_size=None):
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mul(src, index, dim=-1, out=None, dim_size=None):
    index = broadcast(index, src, dim)
    if out is None:
        out = torch.ones(_out_size(src, index, dim, dim_size), dtype=src.dtype, device=src.device)
    return out.scatter_reduce_(dim, index, src, reduce="prod", include_self=True)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)
    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1
    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out


def _scatter_extreme(src, index, dim, out, dim_size, reduce):
    """Group max/min values only. `index` must already be broadcast; empty groups give 0."""
    if out is None:
        out = torch.zeros(_out_size(src, index, dim, dim_size), dtype=src.dtype, device=src.device)
        return out.scatter_reduce_(dim, index, src, reduce=reduce, include_self=False)
    return out.scatter_reduce_(dim, index, src, reduce=reduce, include_self=True)


def _scatter_arg_extreme(src, index, dim, out, dim_size, reduce):
    index = broadcast(index, src, dim)
    if dim < 0:
        dim = src.dim() + dim
    out = _scatter_extreme(src, index, dim, out, dim_size, reduce)
    # arg: smallest position along `dim` whose value equals the group extreme.
    n = src.size(dim)
    shape = [1] * src.dim()
    shape[dim] = n
    pos = torch.arange(n, device=src.device).view(shape).expand_as(src)
    hit = src == out.gather(dim, index)
    cand = torch.where(hit, pos, torch.full_like(pos, n))
    arg = torch.full(out.size(), n, dtype=torch.long, device=src.device)
    arg.scatter_reduce_(dim, index, cand, reduce="amin", include_self=True)
    return out, arg


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return _scatter_arg_extreme(src, index, dim, out, dim_size, "amax")


def scatter_min(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return _scatter_arg_extreme(src, index, dim, out, dim_size, "amin")


def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    if reduce in ("sum", "add"):
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == "mul":
        return scatter_mul(src, index, dim, out, dim_size)
    if reduce == "mean":
        return scatter_mean(src, index, dim, out, dim_size)
    if reduce == "min":
        return scatter_min(src, index, dim, out, dim_size)[0]
    if reduce == "max":
        return scatter_max(src, index, dim, out, dim_size)[0]
    raise ValueError(f"unknown reduce {reduce!r}")


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                    dim_size: Optional[int] = None) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError("`scatter_softmax` can only be computed over tensors with floating point data types.")
    index = broadcast(index, src, dim)
    max_value_per_index = _scatter_extreme(src, index, dim, None, dim_size, "amax")
    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_()
    sum_per_index = scatter_sum(recentered_scores_exp, index, dim, dim_size=dim_size)
    normalizing_constants = sum_per_index.gather(dim, index)
    return recentered_scores_exp.div(normalizing_constants)


def scatter_log_softmax(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                        eps: float = 1e-12, dim_size: Optional[int] = None) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError("`scatter_log_softmax` can only be computed over tensors with floating point data types.")
    index = broadcast(index, src, dim)
    max_value_per_index = _scatter_extreme(src, index, dim, None, dim_size, "amax")
    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered_scores = src - max_per_src_element
    sum_per_index = scatter_sum(src=recentered_scores.exp(), index=index, dim=dim, dim_size=dim_size)
    normalizing_constants = sum_per_index.add_(eps).log_().gather(dim, index)
    return recentered_scores.sub_(normalizing_constants)
