import math
import numpy as np
import torch

def get_batch(dataset, batch_size: int, context_length: int, device: str):
    end = len(dataset) - context_length
    starts = np.random.randint(end, size=batch_size)
    offset = np.arange(context_length)
    x_idx = starts[:, None] + offset[None, :]
    batch = torch.from_numpy(dataset[x_idx]).to(device)
    label = torch.from_numpy(dataset[x_idx+1]).to(device)
    return batch, label


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters

    elif it > cosine_cycle_iters:
        return min_learning_rate

    else:
        lr = min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters)/(cosine_cycle_iters - warmup_iters) * math.pi)) \
             * (max_learning_rate - min_learning_rate)
        return lr


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out
):
    torch.save(
        {"model": model.state_dict(),
         "optimizer": optimizer.state_dict(),
         "iteration": iteration},
        out
    )


def load_checkpoint(
        src,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
):
    ckpt = torch.load(src, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]
