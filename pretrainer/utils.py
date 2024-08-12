import torch
import torch.distributed as dist
from argparse import ArgumentParser
import os

def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)

def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def gather_for_metrics(metric_tensor):
    """
    Gathers and sums metrics across all processes in a distributed training environment.
    
    Args:
    metric_tensor (torch.Tensor): A tensor containing the metric to aggregate.

    Returns:
    torch.Tensor: The aggregated metric.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed package is not initialized")
    
    # Ensure the tensor is on the correct device
    metric_tensor = metric_tensor.to(dtype=torch.float32)

    # Use all_reduce to sum up all the metrics across all processes
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)

    return metric_tensor

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr_warmup_steps", type=int, default=32000)
    parser.add_argument("--output_dir", type=str, default="wav2vec2-indic-voices")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--precision", type=Any[str, int], default=16)
    parser.add_argument('--training_steps', type=int, default=200000)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--gradient_clip_val", type=float, default=8)

    parser.add_argument("--max_gumbel_temperature", type=float, default=2.0)
    parser.add_argument("--min_gumbel_temperature", type=float, default=0.5)
    parser.add_argument("--gumbel_temperature_decay", type=float, default=0.999995)
    parser.add_argument("--save_weights_only", action="store_true")
    parser.add_argument("--save_every_n_steps", type=int, default=10000)
    return parser.parse_args()
