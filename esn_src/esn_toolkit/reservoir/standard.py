"""base implementation of a ESN layer"""
from typing import Callable, Optional, List, Dict

import torch
from torch.distributions import Multinomial

from esn_toolkit.reservoir.topologies import ReservoirTopology, InputTopology, PermutatedReservoirTopology
from .base import BaseReservoir


def _check_float_unit_interval(value, value_name):
    """ Checks if value is a float and in [0,1]. If not raises ValueError
    Args:
        value: value to check
        value_name: the name of the value (to add to the exception
    """
    if value < 0 or type(value) != float or value > 1:
        raise ValueError(f"Invalid value for {value_name}: {value}")


def _check_strictly_positive_number(value, value_name, required_type):
    """ Checks if value is of type required_type and > 0. If not, raises ValueError
    Args:
        value: value to check
        value_name: the name of the value (to add to the exception
    """
    if type(value) != required_type or value <= 0:
        raise ValueError(f"Invalid value for {value_name}: {value}")


class Reservoir(BaseReservoir):
    activation_functions_map = {"tanh": torch.tanh, "sin": torch.sin, "sig": torch.sigmoid}

    def __init__(self, input_size: int, enable_bias: bool, reservoir_size: int, reservoir_connectivity: float = 0.5,
                 reservoir_weight_scale: float = 0.5, input_connectivity: float = 0.5, input_weight_scale: float = 0.5,
                 leaking_rate: float = 0.5, spectral_radius: float = 0.5,
                 available_activation_functions: List[str] = ["tanh"], device="cpu", dry_run=False,
                 reservoir_topology: str = "sparse"):
        """
        A ESN reservoir
        Args:
            input_size: the size of the input
            enable_bias: whether or not to use bias on input
            reservoir_size: the nurber of nodes in the reservoir
            reservoir_connectivity: (number in [0,1])  the rate of connectivity between the reservoir's nodes
            reservoir_weight_scale: (float > 0) the scale of the reservoir weights: [-x, x]
            input_connectivity: (number in [0,1]) the rate of connectivity between the input and the reservoir's nodes.
            input_weight_scale: (float > 0) scale of reservoir weights
            leaking_rate: (number in [0,1]) the leaking rate of this reservoir's nodes
            spectral_radius: spectral radius to conform the reservoir to
            available_activation_functions: unit activation functions to sample from
            device: device to use
            dry_run: if true does not generate the weight matrices
            reservoir_topology: if "sparse" generates a random reservoir topology. if "permutated", will generate a
            permutated reservoir weight matrix (see
            :func:`~esn_toolkit.reservoir.topologies.PermutatedReservoirTopolgy.create_random`). In this case, the
            reservoir_connectivity and reservoir weight scale parameters will be ignored. The spectral radius will be
            used to scale the network
        """
        super().__init__(device)

        self.bias = enable_bias

        _check_strictly_positive_number(input_size, "input_size", int)
        self._input_size = input_size

        _check_strictly_positive_number(reservoir_size, "reservoir_size", int)
        self._reservoir_size = reservoir_size

        _check_float_unit_interval(reservoir_connectivity, "reservoir_connectivity")
        self.reservoir_connectivity = reservoir_connectivity

        _check_strictly_positive_number(reservoir_weight_scale, "reservoir_weight_scale", float)
        self.reservoir_weight_scale = reservoir_weight_scale

        _check_float_unit_interval(input_connectivity, "input_connectivity")
        self.input_connectivity = input_connectivity

        _check_strictly_positive_number(input_weight_scale, "input_weight_scale", float)
        self.input_weight_scale = input_weight_scale

        _check_float_unit_interval(leaking_rate, "leaking_rate")
        self.leaking_rate = leaking_rate

        _check_strictly_positive_number(spectral_radius, "spectral_radius", float)
        self.spectral_radius = spectral_radius

        self.reservoir_weight_matrix = None
        """a (reservoir_size x reservoir_size) matrix"""

        self.input_weight_matrix = None
        """a (reservoir_size x inputs) matrix"""
        if reservoir_topology not in ["sparse", "permutated"]:
            # topology not supported
            raise ValueError(f"invalid reservoir topology: {reservoir_topology}")
        print(f"reservoir topology is {reservoir_topology}")
        self.reservoir_topology = reservoir_topology

        if not dry_run:
            self._init_reservoir_topology()  # sets reservoir/input_weight_matrix

        # generating activation functions
        self.available_activation_functions: List[Callable] = [Reservoir.activation_functions_map[act] for
                                                               act in available_activation_functions]
        self.available_activation_functions_str: List[str] = available_activation_functions

        if not dry_run:
            """Stores a mask for each activation function in the form of a (reservoir_size x 1 x num_activation_func) 
            matrix"""
            self.activation_functions_map: torch.Tensor = self._init_act_functions(self.reservoir_size,
                                                                                   self.available_activation_functions)

        """unit activations. Shape of (reservoir_size x 1)"""
        self.activation_states: Optional[torch.Tensor] = None
        self.reset_reservoir_state()  # this method will set the attribute

        # making sure weights are stored in the correct device
        self.to(device)

    @staticmethod
    def init_from_dict(param_dict) -> 'Reservoir':
        """
        Initializes an instance of the reservoir with hyperparameters parameters taken from param_dict. An additional
        key "type" provides the type of the reservoir. If it does not match raises an error
        Args:
            param_dict: a dictionary with parameters (same keys as __init__)

        Returns: a reservoir instance


        """
        param_dict = param_dict.copy()
        if param_dict["type"] != "Reservoir":
            raise TypeError(f"provided parameters for type {param_dict['type']}")
        del param_dict["type"]
        return Reservoir(**param_dict)

    def param_dict(self) -> Dict:
        """
        Generates a dictionary with the Reservoir's hyperparameters
        Returns: a dictionary with parameters

        """

        p_dict = {"type": "Reservoir",
                  "input_size": self._input_size, "enable_bias": self.bias, "reservoir_size": self._reservoir_size,
                  "input_weight_scale": self.input_weight_scale, "input_connectivity": self.input_connectivity,
                  "spectral_radius": self.spectral_radius, "reservoir_connectivity": self.reservoir_connectivity,
                  "reservoir_weight_scale": self.reservoir_weight_scale, "leaking_rate": self.leaking_rate,
                  "available_activation_functions": self.available_activation_functions_str, "device": self.device,
                  "reservoir_topology": self.reservoir_topology}
        return p_dict

    def __repr__(self):
        params = [f"{k}={v}" for k, v in self.param_dict().items()]

        return f"Reservoir({', '.join(params)})"

    def to(self, device) -> 'Reservoir':
        """
        Changes the reservoir's device inplace.
        Args:
            device: the device to store pytorch tensors to

        Returns: returns the same instance of the reservoir

        """

        if self.device == device:
            return self
        print(f"called to device of std esn with current device {self.device} -> {device}")
        self.reservoir_weight_matrix = self.reservoir_weight_matrix.to(device)
        self.input_weight_matrix = self.input_weight_matrix.to(device)
        self.activation_functions_map = self.activation_functions_map.to(device)
        self.activation_states = self.activation_states.to(device)
        self._device = device
        return self

    @property
    def reservoir_size(self) -> int:
        return self._reservoir_size

    @property
    def input_size(self) -> int:
        return self._input_size

    @staticmethod
    def _init_act_functions(reservoir_size: int, available_activation_functions: List[Callable]) -> torch.Tensor:
        """
        Initializes the activation functions for the reservoir by sampling from available_activation_functions

        Args:
            reservoir_size: the size of the reservoir
            available_activation_functions: a list of activation functions

        Returns: a  (reservoir_size x 1 x num_activation_func) map matrix where each line contains one 1 valued cell
            and 0 otherwise
        """
        # creates a uniform multinomial distribution over activation functions for all of the reservoir's units
        act_prob = torch.ones(reservoir_size, 1, len(available_activation_functions)) / len(
            available_activation_functions)
        # the innermost dimension of act_prob spans the different activation functions
        m = Multinomial(total_count=1, probs=act_prob, validate_args=True)
        return m.sample()

    def _init_reservoir_topology(self):
        if self.reservoir_topology == "sparse":
            print("sparse reservoir")
            self.reservoir_weight_matrix = ReservoirTopology.create_random(self.reservoir_size,
                                                                           self.reservoir_weight_scale,
                                                                           self.reservoir_connectivity,
                                                                           spectral_radius=self.spectral_radius)
        elif self.reservoir_topology == "permutated":
            print("permutated reservoir")
            self.reservoir_weight_matrix = PermutatedReservoirTopology.create_random(self.reservoir_size,
                                                                                     self.spectral_radius)
        else:
            raise ValueError(f"invalid reservoir topology: {self.reservoir_topology}")
        self.input_weight_matrix = InputTopology.create_random(self.input_size, self.reservoir_size, self.bias,
                                                               self.input_weight_scale, self.input_connectivity)

    def reset_reservoir_state(self):
        """(Re)Initializes reservoir's activations.
        Modifies reservoir.activation_states"""
        self.activation_states = torch.zeros(self.reservoir_size, 1, device=self.device)

    def forward(self, input_vector: torch.Tensor):
        """
        applies an input to the network

        Args:
            input_vector: a row vector with the input (1, input_size)
        Returns:
            activation states vector (1, reservoir_size)
        """

        if input_vector.shape != (1, self.input_size):
            raise ValueError(f"input vector has invalid shape {input_vector.shape}. expected (1, {self.input_size}) ")

        with torch.no_grad():
            # transposing input so the vector is shaped to be a column vector
            input_vector = input_vector.reshape(-1, 1)

            if self.bias:
                # adding constant for bias if it is being used
                input_vector = torch.cat([torch.tensor([[1.0]], device=self._device), input_vector], dim=0)

            # get the latest state
            r_t = self.activation_states

            term1 = self.input_weight_matrix.mm(input_vector)
            term2 = self.reservoir_weight_matrix.mm(r_t)

            activations = torch.zeros((self.reservoir_size, 1), device=self._device)
            # calculate the value of the activation functions using the masks

            for act_inx, act_func in enumerate(self.available_activation_functions):
                # use the mask to conditionally apply the activation functions
                activation = act_func(term1 + term2)
                activations += self.activation_functions_map[:, :, act_inx] * activation

            r_t = (1.0 - self.leaking_rate) * r_t + self.leaking_rate * activations

            # update the latest state
            self.activation_states = r_t
        return r_t.transpose(0, 1)
