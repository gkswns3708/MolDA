"""Gradient norm computation utility for per-group logging."""

from typing import Dict


def compute_grad_norms(optimizer) -> Dict[str, float]:
    """Optimizer의 param group별 L2 gradient norm 계산.

    Args:
        optimizer: torch.optim.Optimizer (param_groups에 "name" key 필요)

    Returns:
        {group_name: l2_norm} dict. gradient가 없는 group은 제외.
    """
    norms: Dict[str, float] = {}
    for group in optimizer.param_groups:
        name = group.get("name", "unnamed")
        grad_norm_sq = 0.0
        count = 0
        for p in group["params"]:
            if p.grad is not None:
                grad_norm_sq += p.grad.data.float().norm(2).item() ** 2
                count += 1
        if count > 0:
            norms[name] = grad_norm_sq ** 0.5
    return norms
