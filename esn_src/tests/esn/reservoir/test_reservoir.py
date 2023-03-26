import pytest

import torch

from esn_toolkit.reservoir.standard import Reservoir


@pytest.fixture
def reservoir_size():
    yield 100


@pytest.fixture
def available_activation_functions():
    yield [torch.tanh, torch.relu, torch.sin]


def test_init_act_functions(reservoir_size, available_activation_functions):
    activation_map = Reservoir._init_act_functions(reservoir_size, available_activation_functions)
    assert activation_map.shape == (reservoir_size, 1, len(available_activation_functions))
    assert torch.all(activation_map.sum(dim=2) == 1.0)
