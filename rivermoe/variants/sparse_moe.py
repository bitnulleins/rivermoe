import abc

import torch
from deep_river.utils.tensor_conversion import dict2tensor

from rivermoe.base import BaseVariant
from rivermoe.utils.misc import module_class_name


class SparseMoE(BaseVariant):
    """
    Implementation of Mixtures of Experts with sparse gating mechanism.
    Source: Shazeer et al. (2017) "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
    """

    def __init__(self, top_k: int, **kwargs):
        super().__init__(**kwargs)
        if top_k > self._n_experts:
            raise ValueError(
                f"Top-K value {top_k} is greater than the number of experts {self._n_experts}"
            )

        self.top_k = top_k
        self.noise_std = torch.nn.Parameter(torch.zeros(self._n_experts))

    @property
    def _name(self):
        """
        Overwrite name method to include top_k
        """
        return f"{module_class_name(self)}(k={self.top_k})"

    def _gating(self, x: dict) -> torch.Tensor:
        """Return noisy Top(k) gate prediction

        Parameters
        ----------
        x : dict
            Input data

        Returns
        -------
        torch.Tensor
            Noisy gate prediction
        """
        x_t = dict2tensor(x, features=self.gate.observed_features, device=self.gate.device)

        logits = self.gate.module(x_t)
        raw_noise = self.noise_std.exp() * logits
        noise_stddev = torch.nn.functional.softplus(raw_noise) + 1e-2

        logits = logits + (torch.randn_like(logits) * noise_stddev)
        logits = torch.nn.functional.softmax(logits, dim=-1)

        top_logits, top_indices = logits.topk(min(self.top_k + 1, self._n_experts), dim=1)
        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gate_weights = zeros.scatter(1, top_k_indices, top_k_gates)

        return gate_weights, top_k_indices.flatten().tolist()

    @abc.abstractmethod
    def _predict_weighted(self, x: dict, gate_weights: torch.tensor) -> torch.Tensor:
        """Return weighted prediction for selected experts

        Parameters
        ----------
        x : dict
            Input data
        gate_weights : torch.tensor
            Gate weights
        """
        pass
