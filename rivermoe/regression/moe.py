from typing import List

import abc

import torch
from deep_river.utils.tensor_conversion import float2tensor
from river import base
from river.base.typing import RegTarget

from rivermoe.base import BaseMixtureOfExpert


class MoERegressor(BaseMixtureOfExpert, base.Regressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for expert in self.experts.values():
            if not isinstance(expert, base.Regressor):
                raise ValueError(f"{expert.__class__.__name__} is not a Regressor")

    def predict_one(self, x: dict) -> RegTarget:
        """Return prediction for single input.
        If number of experts is 1, return prediction of single expert.

        Parameters
        ----------
        x : dict
            Input data

        Returns
        -------
        RegTarget
            Prediction value
        """
        if self._n_experts == 1:
            single_expert = next(iter(self.experts))
            return self.experts[single_expert].predict_one(x)
        if not self._moe_initialized:
            self.initialize_moe(x)

        # Adapt gate input dimensions
        self.gate._adapt_input_dim(x)

        self.gate.module.eval()
        return self._predict(x)

    def learn_one(self, x: dict, y: RegTarget) -> "Regressor":
        """Learn from single input (x, y).
        If number of experts is 1, learn from single expert.

        Parameters
        ----------
        x : dict
            Input data
        y : RegTarget
            Label data

        Returns
        -------
        Regressor
            self
        """
        if self._n_experts == 1:
            single_expert = next(iter(self.experts))
            self.update_stats([1])
            return self.experts[single_expert].learn_one(x, y)
        if not self._moe_initialized:
            self.initialize_moe(x)

        # Adapt gate input dimensions
        self.gate._adapt_input_dim(x)

        self._learn(x, y)
        return self

    @abc.abstractmethod
    def _predict(self, x: dict) -> List[float]:
        """Abstract method for prediction.

        Parameters
        ----------
        x : dict
            Input data

        Returns
        -------
        List[float]
            Prediction values
        """
        pass

    @abc.abstractmethod
    def _learn(self, x: torch.Tensor, y: base.typing.RegTarget):
        """Abstract method for learning.

        Parameters
        ----------
        x : torch.Tensor
            Input data
        y : base.typing.ClfTarget
            Label data
        """
        pass

    def _learn_gate(self, y_pred: torch.Tensor, y: base.typing.RegTarget):
        """Learn gate with prediction and label data.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values from experts
        y : base.typing.RegTarget
            True label data
        """
        y_t = float2tensor(y, device=self.gate.device)
        self.gate.module.train()
        self.gate.optimizer.zero_grad()
        loss = self._loss(y_pred, y_t)
        loss.backward()
        self.gate.optimizer.step()
