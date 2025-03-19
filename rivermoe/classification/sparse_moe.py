from typing import Union

import torch

from rivermoe.classification.moe import MoEClassifier
from rivermoe.utils.tensor_conversion import proba2list
from rivermoe.variants.sparse_moe import SparseMoE


class SparseMoEClassifier(MoEClassifier, SparseMoE):
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
        gate_weights, _ = self._gating(x)
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
        gate_weights, top_k_indices = self._gating(x)
        y_pred = self._predict_weighted(x, gate_weights)
        for i in top_k_indices:
            self.experts[i].learn_one(x, y)
        self._learn_gate(y_pred, y)
        self.update_stats(gate_weights.flatten().tolist())

    def _predict_weighted(self, x: dict, gate_weights: torch.tensor):
        """Return weighted prediction

        Parameters
        ----------
        x : dict
            Input data
        gate_weights : torch.tensor
            Gate weights

        Returns
        -------
        base.typing.RegTarget
            Weighted prediction
        """
        active_experts = (gate_weights[0] > 0).nonzero(as_tuple=True)[0].tolist()
        expert_outputs = torch.zeros(
            self._n_experts,
            max(len(self._observed_classes), 2),  # At least size 2 for binary classification
            dtype=torch.float32,
            device=self.gate.device,
        )

        for idx in active_experts:
            expert_outputs[idx] = torch.Tensor(
                proba2list(self.experts[idx].predict_proba_one(x), self._observed_classes)
            )

        y_pred = (expert_outputs * gate_weights.T).sum(dim=0)
        return y_pred.unsqueeze(0)
