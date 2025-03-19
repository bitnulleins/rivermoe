import abc

import torch
from deep_river.utils.tensor_conversion import dict2tensor

from rivermoe.base import BaseVariant


class SoftMoE(BaseVariant):
    """
    Implementation of Mixtures of Experts with soft gating mechanism.
    Source: Jacobs et al. (1991) "Adaptive Mixtures of Local Experts"
    """

    def _gating(self, x: dict) -> torch.Tensor:
        """Return soft gate prediction

        Parameters
        ----------
        x : dict
            Input data

        Returns
        -------
        torch.Tensor
            Soft gate prediction
        """
        x_t = dict2tensor(x, features=self.gate.observed_features, device=self.gate.device)
        logits = self.gate.module(x_t)
        gate_weights = torch.nn.functional.softmax(logits, dim=-1)
        return gate_weights

    @abc.abstractmethod
    def _predict_weighted(self, x: dict, gate_weights: torch.tensor) -> torch.Tensor:
        """Return weighted prediction

        Parameters
        ----------
        x : dict
            Input data
        gate_weights : torch.tensor
            Gate weights
        """
        pass
