from typing import Dict, Union

import abc

import numpy as np
import torch
from deep_river.utils.tensor_conversion import labels2onehot, output2proba
from ordered_set import OrderedSet
from river import base

# from rivermoe.utils.tensor_conversion import dict2onehot
from river.base.typing import ClfTarget

from rivermoe.base import BaseMixtureOfExpert


class MoEClassifier(BaseMixtureOfExpert, base.Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._observed_classes: OrderedSet[ClfTarget] = OrderedSet([])
        for expert in self.experts.values():
            if not isinstance(expert, base.Classifier):
                raise ValueError(f"{expert.__class__.__name__} is not a Classifier")

    def predict_one(self, x: dict) -> base.typing.ClfTarget | None:
        """Return prediction for single input.
        If number of experts is 1, return prediction of single expert.

        Parameters
        ----------
        x : dict
            Input data

        Returns
        -------
        ClfTarget
            Prediction value
        """
        y_pred = self.predict_proba_one(x)
        if y_pred:
            return max(y_pred, key=y_pred.get)
        return None

    def predict_proba_one(self, x: dict) -> Dict[base.typing.ClfTarget, float]:
        """Return prediction probabilities for single input.
        If number of experts is 1, return prediction of single expert.

        Parameters
        ----------
        x : dict
            Input data

        Returns
        -------
        Dict[base.typing.ClfTarget, float]
            Prediction probabilities per class
        """
        if self._n_experts == 1:
            single_expert = next(iter(self.experts))
            return self.experts[single_expert].predict_proba_one(x)
        if not self._moe_initialized:
            self.initialize_moe(x)

        # Adapt gate input dimensions
        self.gate._adapt_input_dim(x)

        self.gate.module.eval()
        y_pred = self._predict(x)
        return output2proba(y_pred, self._observed_classes, self.gate.output_is_logit)[0]

    def learn_one(self, x: dict, y: base.typing.ClfTarget) -> None:
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
        Classifier
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

        # Update observed classes
        self._update_observed_classes(y)

        self._learn(x, y)
        return self

    @abc.abstractmethod
    def _predict(self, x: dict) -> torch.Tensor:
        """Abstract method for prediction.

        Parameters
        ----------
        x : dict
            Input data
        """
        pass

    @abc.abstractmethod
    def _learn(self, x: torch.Tensor, y: base.typing.ClfTarget):
        """Abstract method for learning.

        Parameters
        ----------
        x : torch.Tensor
            Input data
        y : base.typing.ClfTarget
            Label data
        """
        pass

    def _learn_gate(self, y_pred: torch.Tensor, y: base.typing.ClfTarget):
        """Learn gate with prediction and label data.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values from experts
        y : base.typing.ClfTarget
            True label data
        """
        n_classes = y_pred.shape[-1]
        y_onehot = labels2onehot(
            y=y,
            classes=self._observed_classes,
            n_classes=n_classes,
            device=self.gate.device,
        )
        self.gate.module.train()
        self.gate.optimizer.zero_grad()
        loss = self._loss(y_pred, y_onehot)
        loss.backward()
        self.gate.optimizer.step()

    def _update_observed_classes(self, y: base.typing.ClfTarget) -> bool:
        """Update global observed classes over all experts with new class.

        Parameters
        ----------
        y : base.typing.ClfTarget
            Label data

        Returns
        -------
        bool
            New class added or not
        """
        n_existing_classes = len(self._observed_classes)
        if isinstance(y, Union[ClfTarget, np.bool_]):
            self._observed_classes.add(y)
        else:
            self._observed_classes |= y

        # Sort values after adding
        if len(self._observed_classes) > n_existing_classes:
            self._observed_classes = OrderedSet(sorted(self._observed_classes))
            return True
        else:
            return False
