from typing import Union

import torch

from rivermoe.regression.moe import MoERegressor
from rivermoe.variants.sparse_moe import SparseMoE


class SparseMoERegressor(MoERegressor, SparseMoE):
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
        gate_weights, top_k_indices = self._gating(x)
        y_pred = self._predict_weighted(x, gate_weights)
        return y_pred.item()

    def _learn(self, x: dict, y):
        """Learn selected Top(k) experts and return weighted prediction for gate training with loss

        Parameters
        ----------
        x : dict
            Input data
        y : base.typing.RegTarget
            Label data
        """
        gate_weights, top_k_indices = self._gating(x)
        for i in top_k_indices:
            self.experts[i].learn_one(x, y)
        y_pred = self._predict_weighted(x, gate_weights)

        self._learn_gate(y_pred, y)
        self.update_stats(gate_weights.flatten().tolist())

    def _predict_weighted(self, x: dict, gate_weights: torch.tensor) -> torch.Tensor:
        """Return weighted prediction for Top(k) experts

        Parameters
        ----------
        x : dict
            Input data
        gate_weights : torch.tensor
            Gate weights

        Returns
        -------
        torch.Tensor
            Weighted prediction
        """
        active_experts = (gate_weights[0] > 0).nonzero(as_tuple=True)[0].tolist()
        expert_outputs = torch.zeros(self._n_experts, dtype=torch.float32, device=self.gate.device)

        for idx in active_experts:
            expert_outputs[idx] = self.experts[idx].predict_one(x)

        y_pred = (expert_outputs * gate_weights).sum(dim=-1).unsqueeze(0)
        return y_pred
