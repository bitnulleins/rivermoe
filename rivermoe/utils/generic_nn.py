from typing import Callable, Dict, Union, cast

import pandas as pd
import torch
import torch.nn as nn
from deep_river.base import DeepEstimator
from deep_river.classification import Classifier
from deep_river.regression import Regressor
from deep_river.utils import get_activation_fn, get_loss_fn, get_optim_fn
from ordered_set import OrderedSet

SUPPORTED_LAYERS = {
    "linear": nn.Linear,
    "dropout": nn.Dropout,
    "rnn": nn.RNN,
    "lstm": nn.LSTM,
}

OUTPUT_ACTIVATIONS = {
    "softmax": nn.Softmax(dim=-1),
    "sigmoid": nn.Sigmoid(),
}


class GenericNNArchitecture(nn.Module):
    """
    A generic neural network architecture class that allows for flexible configuration
    of input, hidden, and output layers using a tuple of string-based layer definitions.
    """

    def __init__(
        self,
        n_features: int,
        output_dim: int = None,
        layer_configs: list = [],
        output_activation=None,
        activations=None,
    ):
        """
        Parameters
        ----------
        n_features : int
            The dimension of the input features.
        output_dim : int
            The dimension of the output layer.
        layer_configs : list of tuples
            A list where each element is a tuple specifying a layer type (as a string)
            and its parameters. If only a tuple of numbers is provided, it defaults to Linear layers.
            Example: [("linear", 128), ("dropout", 0.2), ("lstm", 64)] or [128, 64].
        output_activation : str, optional
            The activation function to apply to the output layer.
        activations : list of str or str, optional
            A list of activation functions to apply after each layer, if applicable.

        """
        super(GenericNNArchitecture, self).__init__()
        if isinstance(activations, str):
            activations = [activations for _ in range(len(layer_configs) + 1)]
        activations = [
            get_activation_fn(activation)() if activation is not None else None
            for activation in activations
        ]
        self.activations = activations

        self.layers = nn.ModuleList()
        current_dim = n_features

        for idx, config in enumerate(layer_configs):
            # Default to Linear layer if config is a single integer
            if isinstance(config, int):
                config = ("linear", config)

            # Extract layer type and parameters
            layer_type, *layer_params = config
            layer_type = layer_type.lower()  # Normalize to lowercase

            if layer_type not in SUPPORTED_LAYERS:
                raise ValueError(f"Unsupported layer type: {layer_type}")

            # Dynamically create the layer using the corresponding class
            if layer_type == "linear":
                layer = SUPPORTED_LAYERS[layer_type](current_dim, *layer_params)
                current_dim = layer_params[0]  # Update dimension for Linear layers
            elif layer_type == "dropout":
                # Validate dropout probability
                if len(layer_params) != 1 or not (0 <= layer_params[0] <= 1):
                    raise ValueError(
                        f"Dropout probability must be between 0 and 1, got {layer_params}"
                    )
                layer = SUPPORTED_LAYERS[layer_type](*layer_params)
            elif layer_type in ["rnn", "lstm"]:
                hidden_size = layer_params[0]  # The first parameter is hidden_size
                layer = SUPPORTED_LAYERS[layer_type](
                    input_size=current_dim, hidden_size=hidden_size, *layer_params[1:]
                )
                current_dim = hidden_size  # Update current_dim to hidden_size

            self.layers.append(layer)

            # Add activation function if provided
            if activations and idx < len(activations):
                if activations[idx] is not None:
                    self.layers.append(activations[idx])

        # Add final output layer
        if output_dim:
            self.layers.append(nn.Linear(current_dim, output_dim))

        if activations and len(activations) > len(layer_configs):
            if activations[idx + 1] is not None:
                self.layers.append(activations[idx + 1])

        # Add final activation function if provided
        if output_activation in OUTPUT_ACTIVATIONS:
            self.layers.append(OUTPUT_ACTIVATIONS[output_activation])

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, (nn.RNN, nn.LSTM)):
                x, _ = layer(x)  # Handle RNN/LSTM outputs
            else:
                x = layer(x)
        return x

    @classmethod
    def create(
        cls,
        output_dim: int = None,
        layer_configs: list = [],
        output_activation: str = None,
        activations: list = None,
    ) -> nn.Module:
        """Static method that creates a new instance of nn.Module with the specified parameters.

        Parameters
        ----------
        output_dim : int, optional
            Num of output neurons, by default None
        layer_configs : list, optional
            Configuration for NN layers, by default []
        output_activation : str, optional
            Name of output activation function, by default None
        activations : list, optional
            Activation function between layers, by default None

        Returns
        -------
        nn.Module
            PyTorch NN module
        """
        return lambda n_features: cls(
            n_features=n_features,
            output_dim=output_dim,
            layer_configs=layer_configs,
            output_activation=output_activation,
            activations=activations,
        )


class GenericNN(DeepEstimator):
    """
    A generic neural network class that defines the architecture, optimizer, and loss function.
    """

    def __init__(
        self,
        layer_configs: Union[list[int], list[tuple]],
        loss_fn: Union[str, Callable],
        optimizer_fn: Union[str, Callable],
        activation_fn: Union[str, list[Union[Callable, str]]] = None,
        output_dim: int = None,
        output_activation: str = None,
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        self.module_cls = GenericNNArchitecture
        self.module: torch.nn.Module = cast(torch.nn.Module, None)
        self.layer_configs = layer_configs
        self.activation_fn = activation_fn
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.loss_func = get_loss_fn(loss_fn)
        self.loss_fn = loss_fn
        self.optimizer_func = get_optim_fn(optimizer_fn)
        self.optimizer_fn = optimizer_fn
        self.is_feature_incremental = is_feature_incremental
        self.is_class_incremental: bool = False
        self.observed_features: OrderedSet[str] = OrderedSet([])
        self.lr = lr
        self.device = device
        self.kwargs = kwargs
        self.seed = seed
        self.input_layer = cast(torch.nn.Module, None)
        self.input_expansion_instructions = cast(Dict, None)
        self.output_layer = cast(torch.nn.Module, None)
        self.output_expansion_instructions = cast(Dict, None)
        self.module_initialized = False
        torch.manual_seed(seed)

    def initialize_module(self, x: dict | pd.DataFrame, **kwargs):
        del self.kwargs["module"]  # Remove module, if exist
        kwargs = {
            "output_dim": self.output_dim,
            "layer_configs": self.layer_configs,
            "activations": self.activation_fn,
            "output_activation": self.output_activation,
            **self.kwargs,
        }
        super().initialize_module(x=x, **kwargs)


class GenericNNClassifier(GenericNN, Classifier):
    """
    A NN classifier that extends the GenericNN class and adds support for classification tasks.
    """

    def __init__(
        self,
        layer_configs: Union[list[int], list[tuple]],
        loss_fn: Union[str, Callable],
        optimizer_fn: Union[str, Callable],
        output_activation: str = "sigmoid",
        output_dim: int = 2,  # Default for binary classification
        activation_fn: Union[str, list[Union[Callable, str]]] = None,
        lr: float = 1e-3,
        output_is_logit: bool = True,
        is_class_incremental: bool = False,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            layer_configs=layer_configs,
            activation_fn=activation_fn,
            output_dim=output_dim,
            output_activation=output_activation,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            device=device,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            seed=seed,
            **kwargs,
        )
        self.observed_classes: OrderedSet[ClfTarget] = OrderedSet([])
        self.output_is_logit = output_is_logit
        self.is_class_incremental = is_class_incremental


class GenericNNRegressor(GenericNN, Regressor):
    """
    A NN regressor that extends the GenericNN class and adds support for regression tasks.
    """

    def __init__(
        self,
        layer_configs: Union[list[int], list[tuple]],
        loss_fn: Union[str, Callable],
        optimizer_fn: Union[str, Callable],
        activation_fn: Union[str, list[Union[Callable, str]]] = None,
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            layer_configs=layer_configs,
            activation_fn=activation_fn,
            output_dim=1,  # Default for single output regression
            output_activation=None,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            device=device,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            seed=seed,
            **kwargs,
        )
