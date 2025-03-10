import math

from river import base

from rivermoe.regression.sparse_moe import SparseMoERegressor


class SAMoE(SparseMoERegressor):
    """
    Implementation of Mixtures of Experts for regression tasks with SAMoE gating mechanism.
    Source: Dohrn (2025) "Adaptive Machine Learing with Mixtures of Experts"
    """

    drift_detector: base.DriftDetector
    _experts_catalog: dict
    _mean_entropy: float = 1
    _lambda: float = 0

    def __init__(
        self, drift_detector: base.DriftDetector = None, entropy_lambda: float = 0, **kwargs
    ):
        super().__init__(**kwargs)
        self._lambda = entropy_lambda

    def update_stats(self, weights: list, exp: list) -> list:
        super().update_stats(weights, exp)
        entropy = -sum([w * math.log(w) for w in weights])
        all = sum(self._abs_freq.values())
        self._mean_entropy += (entropy - self._mean_entropy) / all

    def add_expert(self, expert: base.Estimator, idx: int = None):
        if idx is None:
            idx = max(self.experts.keys()) + 1
        self.experts[idx] = expert

        if self.gate.module_initialized:
            self._adapt_gate_output_dim(idx)

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
                init_fn=torch.nn.init.kaiming_uniform_,
            )
