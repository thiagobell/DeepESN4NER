"""Defines an interface for reservoirs"""
from typing import Union, Optional, Dict, Any

import torch


class InsufficientLengthError(Exception):
    pass


class BaseReservoir(torch.nn.Module):
    """Defines an interface a Reservoir must implement"""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self._device = device

    def forward(self, input_vec) -> torch.Tensor:
        pass

    @property
    def device(self):
        return self._device

    def to(self, device) -> 'BaseReservoir':
        raise NotImplementedError

    def reset_reservoir_state(self):
        pass

    def embed_sequence(self, sequence: torch.Tensor, transient: Union[float, int], transient_with_zero: bool,
                       reset_states=True) ->\
            torch.Tensor:
        """
        Takes a sequence of embeddings (from a sentence, for example) and embeds it with the esn
        Args:
            sequence: a (n x d) matrix where n is the length of the sequence and d is the input embedding dimension
            transient: the size of the transient as an absolute length if int or ratio if float.
            transient_with_zero: if set, during the transient phase, activations with zeros will be added to the
                activations tensor
            reset_states: if true, reservoir states will be reset before embedding the sentence
        Returns: a (n x D) matrix where n is the number of samples and D is the Reservoir embedding dimension

        """
        if sequence.shape[1] != self.input_size:
            raise ValueError(f"provided sequence has input embedding dimension of {sequence.shape[1]} instead of"
                             f" {self.input_size}")

        if reset_states:
            self.reset_reservoir_state()

        if type(transient) == float:
            tr = int(transient * sequence.shape[0])
        else:
            tr = transient
            if sequence.shape[0] <= tr:
                raise InsufficientLengthError(f"transient of {tr} is too large for sequence with length "
                                              f"{sequence.shape[0]}")

        activations = []
        sequence_device = sequence.device
        if self._device != sequence_device:
            sequence = sequence.to(self._device)

        for element_inx in range(sequence.shape[0]):
            element = sequence[element_inx, :].reshape(1, -1)
            act = self.forward(element)
            if element_inx < tr and transient_with_zero:
                # warmup phase
                zt = torch.zeros((1, self.reservoir_size))
                activations.append(zt)
            else:
                activations.append(act)
        embeddings = torch.cat(activations, dim=0)
        if self._device != sequence_device:
            embeddings = embeddings.to(sequence_device)
        return embeddings

    @property
    def reservoir_size(self) -> int:
        raise NotImplementedError

    @property
    def input_size(self) -> int:
        raise NotImplementedError

    def cpu(self):
        return self.to("cpu")

    def cuda(self, gpu_number: Optional[int] = None):
        if gpu_number is not None:
            return self.to(f"cuda:{gpu_number}")
        return self.to("cuda")

    @staticmethod
    def init_from_dict(param_dict) -> 'BaseReservoir':
        """
        Initializes an instance of the reservoir with hyperparameters parameters taken from param_dict
        Args:
            param_dict: a dictionary with parameters (same keys as __init__)

        Returns: a reservoir instance

        """
        raise NotImplementedError

    def param_dict(self) -> Dict[str, Any]:
        """
        Generates a dictionary with the Reservoir's hyperparameters
        Returns: a dictionary with parameters

        """
        raise NotImplementedError

