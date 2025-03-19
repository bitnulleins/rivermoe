from typing import Union

import torch
from river import base

from rivermoe.regression.moe import MoERegressor
from rivermoe.variants.soft_moe import SoftMoE


class SoftMoERegressor(MoERegressor, SoftMoE):
    def _predict(self, x: Union[torch.Tensor, dict]) -> base.typing.RegTarget:
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
        return y_pred.item()

    def _learn(self, x: dict, y: base.typing.RegTarget):
        """Learn all experts and return weighted prediction for gate training with loss

        Parameters
        ----------
        x : dict
            Input data
        y : base.typing.RegTarget
            Label data
        """
        gate_weights = self._gating(x)
        [expert.learn_one(x, y) for expert in self.experts.values()]
        y_pred = self._predict_weighted(x, gate_weights)

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
        torch.tensor
            Weighted prediction
        """
        expert_outputs = [self.experts[i].predict_one(x) for i in self.experts.keys()]
        expert_outputs = torch.tensor(expert_outputs, dtype=torch.float32, device=self.gate.device)
        y_pred = (expert_outputs * gate_weights).sum(dim=-1).unsqueeze(0)
        return y_pred
