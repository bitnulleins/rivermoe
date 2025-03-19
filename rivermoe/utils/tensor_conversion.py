import torch
from ordered_set import OrderedSet


def proba2list(preds: dict, classes: OrderedSet) -> list:
    """
    Convert a dictionary to a tensor.

    Parameters
    ----------
    preds : dict
        Dictionary.
    classes : OrderedSet
        Set of possible clases.

    Returns
    -------
    list
        Probabilities as list
    """
    num_classes = len(classes)

    if num_classes == 0:
        return [0.5, 0.5]

    values = []
    for cls in classes:
        values.append(preds.get(cls, 0.0))

    if num_classes < 2:
        p = values[0] if values else 0.5
        values.append(1 - p)

    return values


def proba2tensor(
    preds: dict,
    classes: OrderedSet,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert a dictionary to a tensor.

    Parameters
    ----------
    preds : dict
        Predictions
    classes : OrderedSet
        Set of possible clases
    device: str
        Device
    dtype: torch.dtype
        Dtype

    Returns
    -------
        torch.Tensor
    """
    return torch.tensor([proba2list(preds, classes)], dtype=dtype, device=device)
