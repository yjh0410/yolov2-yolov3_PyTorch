# from github: https://github.com/ruinmessi/ASFF/blob/master/utils/distributed_util.py

import torch


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def synchronize():
    """
    Helper function to synchronize between multiple processes when
    using distributed training
    """
    if not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if world_size == 1:
        return

    def _send_and_wait(r):
        if rank == r:
            tensor = torch.tensor(0, device="cuda")
        else:
            tensor = torch.tensor(1, device="cuda")
        torch.distributed.broadcast(tensor, r)
        while tensor.item() == 1:
            time.sleep(1)

    _send_and_wait(0)
    # now sync on the main process
    _send_and_wait(1)


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            if len(loss_dict[k].size()) == 0:
                all_losses.append(loss_dict[k].unsqueeze(0))
            else:
                all_losses.append(loss_dict[k])
        
        all_losses = torch.stack(all_losses, dim=0)
        torch.distributed.reduce(all_losses, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses