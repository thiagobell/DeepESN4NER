"""This module implements the different weight topologies anc configurations for ESN reservoirs"""

from typing import Optional

import torch
from torch.distributions.uniform import Uniform
from torch.distributions.binomial import Binomial

class ReservoirTopology:
    @classmethod
    def create_random(cls, size, scaling=0.5, connectivity=1.0, fixed_weight: Optional[float] = None,
                      spectral_radius: Optional[float] = None) -> torch.Tensor:
        """
        Creates a topology in the form of a randomly connected reservoir.
        Args:
            size: reservoir size
            scaling: the scaling of the weights. a scale of x implies the range of weight values is [-x, x]
            connectivity: ratio of connectivity in the graph. i.e. 5% (0.05) implies only 5% of connected
              edges exist.
            fixed_weight: if set, all weights will be valued fixed_weight or zero.
            spectral_radius: if not none, scales the matrix so its spectal radius matches this parameter

        Returns: a reservoir weight matrix in the form of an adjacency matrix (with weights) where the jth row
         contains the input weights of the jth neuron. A connection from i -> j with weight 1.0 means that
          weight_matrix[j, i] = 1

        """
        mask_distribution = Binomial(total_count=1, probs=connectivity)
        if fixed_weight is None:
            # samples weights from a uniform distribution
            reservoir_weights = Uniform(-scaling, scaling).sample((size, size))
            if connectivity < 1.0:
                # calculates a mask to remove edges

                # use a binomial distribution to generate a connectivity mask

                edge_mask = mask_distribution.sample((size, size))
                reservoir_weights = reservoir_weights.mul(edge_mask)
        else:
            # network should have only have weights with a fixed value (fixed_weight)
            reservoir_weights: torch.Tensor = (mask_distribution.sample((size, size))*fixed_weight).float()

        if spectral_radius is not None:
            reservoir_weights = ReservoirTopology.scale_by_spectral_radius(reservoir_weights, spectral_radius)

        return reservoir_weights

    @staticmethod
    def spectral_radius_from_matrix(matrix):
        """ returns the actual spectral radius of the reservoirs weight matrix"""
        return (torch.max(torch.abs(torch.eig(matrix)[0])) + 1e-5).item()

    @staticmethod
    def scale_by_spectral_radius(matrix: torch.tensor, spectral_radius: float) -> torch.tensor:
        """
        Scales the provided matrix so its spectral radius matches the one provided
        Args:
            matrix:
            spectral_radius:

        Returns:

        """
        rad = ReservoirTopology.spectral_radius_from_matrix(matrix)
        weight_m = matrix / rad

        # Force spectral radius
        weight_m = weight_m * spectral_radius
        return weight_m


class PermutatedReservoirTopology:
    @classmethod
    def create_random(cls, size, weight: float) -> torch.Tensor:
        """
        Creates a topology by permutating the columns of the identity matrix
        Args:
            size: reservoir size
            weight: the weight of the connections, will correspond to the spectral radius

        Returns: a reservoir weight matrix in the form of an adjacency matrix (with weights) where the jth row
         contains the input weights of the jth neuron. A connection from i -> j with weight 1.0 means that
          weight_matrix[j, i] = 1

        """
        weights = torch.eye(size, requires_grad=False).float() * torch.tensor(weight)
        permutation = torch.randperm(size)
        weights = weights[:, permutation]
        return weights

class InputTopology:
    @classmethod
    def create_random(cls, input_size, reservoir_size, bias=False, scaling=0.5, connectivity=1.0) -> torch.Tensor:
        """
        Creates a random Reservoir input topology

        Args:
            input_size: size of input
            reservoir_size: size of the reservoir
            bias:  determines if bias is included
            scaling: scaling of weights [-scaling, scaling]
            connectivity: the ratio of connections to be enabled

        Returns:  a (reservoir_size x inputs) matrix with weights of the input connections. If a bias is present,
         it is placed in the first column.

        """
        weight_matrix = cls._generate_weight_matrix(input_size, reservoir_size, bias, scaling, connectivity)
        return weight_matrix

    @classmethod
    def _generate_weight_matrix(cls, input_size, reservoir_size, bias=False, scaling=0.5, connectivity=1.0):
        effective_input_size = input_size
        if bias:
            effective_input_size += 1

        # samples weights from a uniform distribution
        input_weights = Uniform(-scaling, scaling).sample((reservoir_size, effective_input_size))

        # use a binomial distribution to generate a connectivity mask
        mask_distribution = Binomial(total_count=1, probs=connectivity)
        edge_mask = mask_distribution.sample((reservoir_size, effective_input_size))
        return input_weights.mul(edge_mask)
