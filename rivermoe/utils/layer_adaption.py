from typing import Callable

import torch
import torch.nn as nn
from deep_river.utils.layer_adaptation import SUPPORTED_LAYERS


def reduce_layer(
    layer: nn.Module,
    keep_neurons: list[int],
    instructions: dict | None = None,
    init_fn: Callable = nn.init.normal_,
):
    """
    Modifies a PyTorch layer to keep only the specified neurons and updates the output size.

    Parameters
    ----------
    layer : nn.Module
        The PyTorch layer to be modified (e.g., nn.LSTM, nn.Linear).
    keep_neurons : list[int]
        Indices of the neurons to retain.
    instructions : dict, optional
        Instructions for modifying the layer parameters. If None,
        they will be generated automatically based on the layer type.
    init_fn : Callable, optional
        Initialization function for new weights, by default nn.init.normal_.
    """
    if not isinstance(layer, SUPPORTED_LAYERS):
        raise ValueError(f"Unsupported layer type: {type(layer)}")

    if instructions is None:
        instructions = load_instructions(layer)

    for param_name, instruction in instructions.items():
        param = getattr(layer, param_name)

        if instruction == "output_attribute":
            # Update the output size attribute (e.g., out_features for Linear)
            if hasattr(layer, "out_features"):
                layer.out_features = len(keep_neurons)
            setattr(layer, param_name, len(keep_neurons))

        elif isinstance(instruction, dict):
            for target in ["output"]:
                if target not in instruction:
                    continue
                for axis_info in instruction[target]:
                    axis = axis_info["axis"]
                    n_subparams = axis_info["n_subparams"]

                    # Create tensor for the indices to keep
                    device = param.device if isinstance(param, torch.Tensor) else "cpu"
                    index = torch.tensor(keep_neurons, device=device)

                    # Select only the specified neurons along the output axis
                    chunks = torch.chunk(param, chunks=n_subparams, dim=axis)
                    selected_chunks = [
                        chunk.index_select(dim=axis, index=index) for chunk in chunks
                    ]
                    param = torch.cat(selected_chunks, dim=axis)

        # Check if the parameter is a tensor
        if isinstance(param, torch.Tensor):
            setattr(layer, param_name, nn.Parameter(param))
        else:
            setattr(layer, param_name, param)

    # Special handling for nn.Linear to ensure alignment of out_features
    if isinstance(layer, nn.Linear):
        layer.out_features = len(keep_neurons)
