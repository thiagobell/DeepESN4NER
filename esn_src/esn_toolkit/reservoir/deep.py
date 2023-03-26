"""Defines a deep reservoir as a torch layer"""
from typing import List, Union

import torch

from esn_toolkit.reservoir.standard import Reservoir
from .base import BaseReservoir
from .param import init_reservoir_from_param
from .standard import Reservoir
from .identity import Identity


class DeepReservoir(BaseReservoir):
    allowed_contained_reservoirs = Union[Reservoir, Identity]

    def __init__(self, reservoirs: List[allowed_contained_reservoirs], device: str = "cpu"):
        # should we take a list of ESN layers as input?
        super().__init__(device)
        for rs in reservoirs:
            if type(rs) not in DeepReservoir.allowed_contained_reservoirs.__args__:
                TypeError(f"reservoir of type {type(rs)} is not allowed to be contained by a DeepReservoir")
        self.layers: torch.nn.ModuleList[Reservoir] = torch.nn.ModuleList([rs.to(device) for rs in reservoirs])
        self._input_size = reservoirs[0].input_size

        # the reservoir size is the sum of the size of each reservoir layer
        self._reservoir_size = sum(map(lambda reservoir: reservoir.reservoir_size, reservoirs))

    def to(self, device):
        if self._device == device:
            return self

        for reservoir_inx, reservoir in enumerate(self.layers):
            self.layers[reservoir_inx] = reservoir.to(device)

        self._device = device
        return self

    @property
    def reservoir_size(self) -> int:
        return self._reservoir_size

    @property
    def input_size(self) -> int:
        return self._input_size

    def forward(self, input_vec):
        if input_vec.shape != (1, self.input_size):
            raise ValueError(f"input vector of wrong shape provided. expected (1,{self.input_size}) but "
                             f"received {input_vec.shape}")
        activations = []
        act = input_vec
        for layer in self.layers:
            act = layer(act)
            activations.append(act)

        # creates a single activation vector with the activations of all layers
        return torch.cat(activations, dim=1)

    def reset_reservoir_state(self):
        """(Re)Initializes reservoir's activations.
        Modifies reservoir.activation_states"""
        for l_inx in range(len(self.layers)):
            # taking indirect way so we get type inference from pycharm
            reservoir: Reservoir = self.layers[l_inx]
            reservoir.reset_reservoir_state()

    @staticmethod
    def init_from_dict(param_dict) -> 'DeepReservoir':
        """
        Initializes an instance of the reservoir with hyperparameters parameters taken from param_dict. An additional
        key "type" provides the type of the reservoir. If it does not match raises an error
        Args:
            param_dict: a dictionary with parameters (same keys as __init__)

        Returns: a reservoir instance
        """
        param_dict = param_dict.copy()
        if param_dict["type"] != "DeepReservoir":
            raise TypeError(f"provided parameters for type {param_dict['type']} and not DeepReservoir")
        del param_dict["type"]
        reservoirs = []
        # first initialize layers
        for reservoir_param in param_dict["reservoirs"]:
            # making sure device is the same as the wrapper
            reservoir_param["device"] = param_dict["device"]
            reservoirs.append(init_reservoir_from_param(reservoir_param))

        del param_dict["reservoirs"]
        return DeepReservoir(reservoirs, **param_dict)

    def param_dict(self):
        """
        Generates a dictionary with the Reservoir's hyperparameters
        Returns: a dictionary with parameters

        """

        p_dict = {
            "type": "DeepReservoir",
            "reservoirs": [res.param_dict() for res in self.layers],
            "device": self.device
        }
        return p_dict

    def __repr__(self):
        rep = f"DeepReservoir(" + "\n              ".join([repr(res) for res in self.layers] +
                                                          ["device="+self.device]) + ")"
        return rep

