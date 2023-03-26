"""
This module implements an Identity reservoir (which just replicates the input)
"""
from typing import Dict, Union
import torch

from .base import BaseReservoir


class Identity(BaseReservoir):
    """
    A Identity reservoir which simply replicates its input
    """
    def __init__(self, size):
        super().__init__()
        self._size = size

    @staticmethod
    def init_from_dict(param_dict) -> 'Identity':
        """
        Initializes an instance of the reservoir with hyperparameters parameters taken from param_dict. An additional
        key "type" provides the type of the reservoir. If it does not match raises an error
        Args:
            param_dict: a dictionary with parameters (same keys as __init__)

        Returns: a reservoir instance


        """
        param_dict = param_dict.copy()
        if param_dict["type"] != "Identity":
            raise TypeError(f"provided parameters for type {param_dict['type']}")
        del param_dict["type"]
        return Identity(**param_dict)

    def param_dict(self) -> Dict:
        """
        Generates a dictionary with the Reservoir's hyperparameters
        Returns: a dictionary with parameters

        """

        p_dict = {"type": "Identity",
                  "size": self._size}
        return p_dict

    def __repr__(self):
        params = [f"{k}={v}" for k, v in self.param_dict().items()]

        return f"IdentityReservoir({', '.join(params)})"

    def forward(self, input_vec):
        if input_vec.shape[1] != self._size:
            raise ValueError(f"Invalid shape of input. Expected a {self._size}-dimensional input")

        return input_vec

    def reset_reservoir_state(self):
        pass

    def embed_sequence(self, sequence: torch.Tensor, transient: Union[float, int], transient_with_zero: bool) -> torch.Tensor:
        if sequence.shape[1] != self._size:
            raise ValueError(f"Invalid shape of input. Expected a {self._size}-dimensional input")

        return sequence

    @property
    def reservoir_size(self) -> int:
        return self._size

    @property
    def input_size(self) -> int:
        return self._size

    def to(self, device):
        return self
