from typing import Union

import torch

from rivermoe.classification.moe import MoEClassifier
from rivermoe.utils.tensor_conversion import proba2list
from rivermoe.variants.soft_moe import SoftMoE


class SoftMoEClassifier(MoEClassifier, SoftMoE):
    def _predict(self, x: Union[torch.Tensor, dict]) -> torch.Tensor:
        """Return weighted prediction

        Parameters
        ----------
        x : Union[torch.Tensor, dict]
            Input data

        Returns
        -------
        base.typing.RegTarget
            Weighted prediction
        """
        gate_weights = self._gating(x)
        y_pred = self._predict_weighted(x, gate_weights)
        return y_pred

    def _learn(self, x: dict, y):
        """Learn all experts and return weighted prediction for gate training with loss

        Parameters
        ----------
        x : dict
            Input data
        y : base.typing.RegTarget
            Label data
        """
        gate_weights = self._gating(x)
        y_pred = self._predict_weighted(x, gate_weights)
        [expert.learn_one(x, y) for expert in self.experts.values()]

        self._learn_gate(y_pred, y)
        self.update_stats(gate_weights.flatten().tolist())

    def _predict_weighted(self, x: dict, gate_weights: torch.tensor):
        """Return weighted prediction

        Parameters
        ----------
        x : dict
            Input data
        weights : torch.tensor
            Gate weights
        exptert_indices : list
            Expert indices
        """
        expert_outputs = [
            proba2list(expert.predict_proba_one(x), self._observed_classes)
            for expert in self.experts.values()
        ]
        expert_outputs = torch.tensor(expert_outputs, dtype=torch.float32, device=self.gate.device)
        y_pred = (expert_outputs * gate_weights.T).sum(dim=0)
        return y_pred.unsqueeze(0)
