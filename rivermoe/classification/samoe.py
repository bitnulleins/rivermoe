from river import base

from rivermoe.classification.sparse_moe import SparseMoEClassifier


class SAMoEClassifier(SparseMoEClassifier):
    """Self-Adaptive Mixture of Experts Classifier"""

    def __init__(self, drift_detector: base.DriftDetector, **kwargs):
        super().__init__(top_k=1, **kwargs)
        self.drift_detector = drift_detector
        self._experts_catalog = self.experts.copy()

    def learn_one(self, x: dict, y: base.typing.ClfTarget):
        """Check if drift is detected and then add a new expert.
        After that hook, call default learn_one method.

        Parameters
        ----------
        x : dict
            Input data
        y : base.typing.ClfTarget
            Target value
        """
        y_pred = self.predict_one(x)
        detec_in = 0 if y == y_pred else 1
        self.drift_detector.update(detec_in)
        if self.drift_detector.drift_detected:
            if not self._moe_initialized:
                self.initialize_moe(x)
            expert_index = (self._n_experts - 1) % len(self._experts_catalog)
            selected_expert = self._experts_catalog[expert_index]
            self.add_expert(selected_expert)

        super().learn_one(x, y)

    def add_expert(self, expert: base.Estimator):
        """Add expert to the MoE

        Parameters
        ----------
        expert : base.Estimator
            Expert to add
        """
        idx = max(self.experts.keys()) + 1
        self.experts[idx] = expert
        self._gate_weights[idx] = 0
        self._abs_freq[idx] = 0

        if self.gate.module_initialized:
            self._adapt_gate_output_dim(idx)
