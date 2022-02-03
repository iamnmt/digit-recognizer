import torch


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {k: move_to(v, device) for k, v in obj.items()}
        return res
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to(list(obj), device))
    else:
        raise TypeError("Invalid type for move_to")


def detach(obj):
    if torch.is_tensor(obj):
        return obj.detach()
    elif isinstance(obj, dict):
        res = {k: detach(v) for k, v in obj.items()}
        return res
    elif isinstance(obj, list):
        return [detach(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(detach(list(obj)))
    else:
        raise TypeError("Invalid type for detach")
