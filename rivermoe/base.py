from typing import List, Union

import abc

import deep_river
import torch
import torch.nn as nn
from river import base

from rivermoe.utils.misc import module_class_name

try:
    from graphviz import Digraph
except ImportError as e:
    raise ValueError("You have to install graphviz to use the draw method") from e


class _TestModule(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.dense0 = torch.nn.Linear(n_features, 10)
        self.nonlin = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.dense1 = torch.nn.Linear(10, 5)
        self.output = torch.nn.Linear(5, 1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.softmax(self.output(X))
        return X


class BaseMixtureOfExpert(base.Estimator):
    """Base class for all Mixtures of Experts

    Parameters
    ----------
    gate : deep_river.classification.Classifier
        Neural gate
    experts : Union[base.Estimator,List[base.Estimator]]
        List of expert models
    """

    gate: deep_river.classification.Classifier
    experts: Union[base.Estimator, List[base.Estimator]]

    @classmethod
    def _unit_test_params(cls):
        """
        Returns a dictionary of parameters to be used for unit
        testing the respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing the
            respective class.
        """
        yield {
            "gate": _TestModule,
            "experts": [_TestModule for _ in range(5)],
        }

    @classmethod
    def _unit_test_skips(cls) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases,
        some estimators might not
        be able to pass certain checks.
        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return set()

    def __init__(
        self,
        gate: deep_river.classification.Classifier,
        experts: Union[base.Estimator, List[base.Estimator]],
        seed: int = 42,
    ):
        """Initialize Mixture of Experts

        Parameters
        ----------
        gate : deep_river.classification.Classifier
            Gate of the MoE
        experts : Union[base.Estimator, List[base.Estimator]]
            List of experts (ML or deep learning models)
        seed : int, optional
            random seed, by default 42

        Raises
        ------
        ValueError
            Not initialized component
        ValueError
            Gate has to be of type Classifier
        """
        self.experts = {idx: expert.clone() for idx, expert in enumerate(experts)}
        self.gate = gate.clone()
        self._moe_initialized = False
        self._abs_freq = {idx: 0 for idx in self.experts.keys()}
        self._gate_weights = {idx: 0 for idx in self.experts.keys()}
        self.seed = seed

        for component in [self.gate] + list(self.experts.values()):
            if isinstance(component, type):
                raise ValueError(f"{component.__class__.__name__} is not initialized.")

        if not isinstance(self.gate, deep_river.classification.Classifier):
            raise ValueError(f"Gate has to be of type Classifier.")

    @property
    def _name(self) -> str:
        """Extract name of MoE-class

        Returns
        -------
        str
            Name of MoE
        """
        return module_class_name(self)

    @property
    def _raw_memory_usage_per_component(self) -> int:
        """Memory usage of each expert in bytes per MoE-component.

        Returns
        -------
        int
            Memory in bytes
        """
        memory = {idx: exp._raw_memory_usage for idx, exp in self.experts.items()}
        memory["gate"] = self.gate._raw_memory_usage
        return memory

    @property
    def _rel_freq(self) -> list:
        """Return relative frequency of experts, based on absolute frequency

        Returns
        -------
        list
            Relative frequency of experts
        """
        all = sum(self._abs_freq.values())
        if all == 0:
            return self._abs_freq
        return {idx: num / all for idx, num in self._abs_freq.items()}

    def update_stats(self, weights: list):
        """Update statistics of MoE.
        Extract absolute frequency from weights, if >= 0.

        Parameters
        ----------
        weights : list
            Gate weights for each expert
        """
        for idx, weight in enumerate(weights):
            self._gate_weights[idx] = weight
            if weight > 0:
                self._abs_freq[idx] += 1

        # Normalize weights
        total = sum(self._gate_weights.values())
        if total > 0:
            self._gate_weights = {idx: weight / total for idx, weight in self._gate_weights.items()}

    @property
    def _n_experts(self) -> int:
        """Return number of experts of expert setting

        Returns
        -------
        int
            Number of experts
        """
        return len(self.experts)

    def initialize_moe(self, x: dict) -> None:
        """Initialize Mixture of Experts

        Parameters
        ----------
        x : dict
            First input data

        Raises
        ------
        ValueError
            Same size in output neurons and number of experts
        """
        torch.manual_seed(self.seed)

        self.gate.is_class_incremental = True
        self.gate.is_features_incremental = True
        self.gate.initialize_module(x=x, **self.gate.kwargs)

        if self.gate.output_layer.out_features > len(self.experts):
            raise ValueError(
                "Number of output neurons has to be lower or equal to number of experts"
            )

        for idx in self.experts.keys():
            self._adapt_gate_output_dim(idx)

        def nn_init(layer):
            if isinstance(layer, nn.Linear):
                # Constant weight initialisation, so no prefered order of experts
                torch.nn.init.constant_(layer.weight, 1.0 / self._n_experts)
                # Zero bias for all experts
                torch.nn.init.zeros_(layer.bias)

        self.gate.module.apply(nn_init)

        self._moe_initialized = True

    def _adapt_gate_output_dim(self, y: base.typing.ClfTarget):
        """Adaptiere die Dimensionen des Experten-Outputs

        Parameters
        ----------
        y : base.typing.ClfTarget
            Expert index
        """
        has_new_expert = self.gate._update_observed_classes(y)
        if has_new_expert and self.gate.output_layer.out_features < len(self.gate.observed_classes):
            deep_river.utils.layer_adaptation.expand_layer(
                layer=self.gate.output_layer,
                instructions=self.gate.output_expansion_instructions,
                target_size=len(self.gate.observed_classes),
                output=True,
                init_fn=torch.nn.init.normal_,
            )

    def _loss(
        self, y_pred: torch.Tensor, y_true: Union[base.typing.ClfTarget, base.typing.RegTarget]
    ) -> torch.Tensor:
        """Overwritable loss function for custom loss calculation

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y_true : Union[base.typing.ClfTarget, base.typing.RegTarget]
            True label values

        Returns
        -------
        torch.Tensor
            Loss value
        """
        return self.gate.loss_func(y_pred, y_true)

    def draw(self) -> Digraph:
        """Visualisierung des Mixture of Experts

        Returns
        -------
        Digraph
            Finished digraph
        """

        def _add_expert_nodes(dot, i, exp):
            """Hinzufügen der Expertennodes und ihrer Kombination"""
            dot.node(
                f"Expert_{i}",
                f"Expert {i}:\n{module_class_name(exp)}",
                shape="box",
                style="rounded",
            )
            dot.node(f"Combine_{i}", "×", shape="circle")

        active_experts = {idx: exp for idx, exp in self.experts.items() if exp is not None}
        dot = Digraph(format="png")
        dot.graph_attr.update(
            {
                "margin": "0",
                "pad": "0.02",
                "rankdir": "LR",
            }
        )
        dot.node_attr.update({"fontname": "trebuchet"})

        # Input
        dot.node("Input", "Input", shape="cylinder")

        # Gate
        top_gate_text = f"{{<top>Gate: {module_class_name(self.gate)}}}"
        bottom_gate_text = f"{{<bottom>{self._name}}}"
        dot.node(
            "Gate",
            top_gate_text + "|" + bottom_gate_text,
            shape="record",
            rank="min",
        )

        # Experts
        for i, exp in active_experts.items():
            _add_expert_nodes(dot, i, exp)

        # Results
        dot.node("Combine", "Σ", shape="box")
        dot.node("Output", "Output", shape="ellipse")

        # Edges
        dot.edge("Input", "Gate")
        for i, exp in active_experts.items():
            dot.edge("Input", f"Expert_{i}")
            dot.edge(f"Expert_{i}", f"Combine_{i}")
            dot.edge(f"Gate", f"Combine_{i}", style="dashed")
            dot.edge(f"Combine_{i}", "Combine")

        dot.edge("Combine", "Output")

        # Subgraph for expert pool
        with dot.subgraph(name="cluster_experts") as s:
            s.graph_attr.update({"rank": "same", "margin": "5"})
            for i, exp in active_experts.items():
                if exp is not None:
                    s.node(f"Expert_{i}")

        return dot


class BaseVariant(BaseMixtureOfExpert):
    """
    Base class for all variants of MoE
    """

    @abc.abstractmethod
    def _gating(self, x: dict) -> torch.Tensor:
        """Implementation fo gate strategy of variant

        Parameters
        ----------
        x : dict
            Input data

        Returns
        -------
        torch.Tensor
            Gate weights
        """
        pass
