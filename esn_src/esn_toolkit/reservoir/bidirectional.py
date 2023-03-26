"""
This module implements a wrapper around a class implementing a Reservoir's interface
"""
from typing import Union
import torch
import copy
from .base import BaseReservoir
from .standard import Reservoir
from .identity import Identity
from .deep import DeepReservoir


class Bi(BaseReservoir):
    allowed_contained_reservoirs = Union[Reservoir, Identity, DeepReservoir]
    """
    Creates a bi directional reservoir out of a unidirectional one
    """
    def __init__(self, reservoir: allowed_contained_reservoirs):
        super().__init__()
        if type(reservoir) not in DeepReservoir.allowed_contained_reservoirs.__args__:
            TypeError(f"reservoir of type {type(reservoir)} is not allowed to be contained by a Bidirectional reservoir")
        # we make copies of the reservoir to have meaningful latent contexts between sentences
        self.forward_reservoir = copy.deepcopy(reservoir)
        self.backward_reservoir = copy.deepcopy(reservoir)

    def reset_reservoir_state(self):
        self.forward_reservoir.reset_reservoir_state()
        self.backward_reservoir.reset_reservoir_state()

    def forward(self, x):
        raise NotImplementedError("forwarded method not implemented by bidirectional reservoir")

    def to(self, device):
        self.backward_reservoir = self.backward_reservoir.to(device)
        self.forward_reservoir = self.forward_reservoir.to(device)
        return self

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
            reset_states: if true, reservoir states will be reset before embedding the sentence (including between directions)
        Returns: a (n x 2D) matrix where n is the number of samples and D is the Reservoir embedding dimension.
        The second dimension contains the forward then the backward dimensions.
        """
        forward_embd = self.forward_reservoir.embed_sequence(sequence, transient, transient_with_zero, reset_states)
        # reverting the order of the sequence
        inv_seq = sequence.flip(0)
        backwd_embd = self.backward_reservoir.embed_sequence(inv_seq, transient, transient_with_zero, reset_states)
        # need to flip embeddings back into the order of sequence
        backwd_embd = backwd_embd.flip(0)

        return torch.cat([forward_embd, backwd_embd], dim=1)

    @property
    def reservoir_size(self) -> int:
        return self.forward_reservoir.reservoir_size*2

    @staticmethod
    def init_from_dict(param_dict) -> 'Bi':
        """
        Initializes an instance of the reservoir with hyperparameters parameters taken from param_dict. An additional
        key "type" provides the type of the reservoir. If it does not match raises an error
        Args:
            param_dict: a dictionary with parameters (same keys as __init__)

        Returns: a reservoir instance
        """
        # importing here to avoid import loops
        from esn_toolkit.reservoir.param import init_reservoir_from_param
        param_dict = param_dict.copy()
        if param_dict["type"] != "Bi":
            raise TypeError(f"provided parameters for type {param_dict['type']} and not DeepReservoir")
        del param_dict["type"]
        # initialize reservoir

        # making sure device is the same as the wrapper
        param_dict["reservoir"]["device"] = param_dict["device"]
        reservoir = init_reservoir_from_param(param_dict["reservoir"])
        del param_dict["reservoir"]
        return Bi(reservoir)

    def param_dict(self):
        """
        Generates a dictionary with the Reservoir's hyperparameters
        Returns: a dictionary with parameters

        """

        p_dict = {
            "type": "Bi",
            "reservoir": self.forward_reservoir.param_dict(),
            "device": self.device
        }
        return p_dict

    def __repr__(self):
        rep = f"Bi(" + repr(self.forward_reservoir) + ")"
        return rep
