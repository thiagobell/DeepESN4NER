import pytest

import torch

from esn_toolkit.reservoir import Reservoir
from esn_toolkit.reservoir import DeepReservoir

@pytest.fixture
def available_activation_functions():
    yield [torch.tanh, torch.relu, torch.sin]



@pytest.fixture
def deep_reservoir(available_activation_functions) -> DeepReservoir:
    res = [
        Reservoir(120, True, 100, 0.32, 0.12, 1.0, 1.0, 0.23, 0.79, available_activation_functions),
        Reservoir(100, False, 300, 0.32, 0.12, 1.0, 1.0, 0.23, 0.79, available_activation_functions),
        Reservoir(300, False, 500, 0.32, 0.12, 1.0, 1.0, 0.23, 0.79, available_activation_functions),
        Reservoir(500, True, 100, 0.32, 0.12, 1.0, 1.0, 0.23, 0.79, available_activation_functions)
    ]
    dr = DeepReservoir(res)

    yield dr


def test_deep_reservoir_initialized(deep_reservoir):
    assert type(deep_reservoir) == DeepReservoir


def test_forward(deep_reservoir: DeepReservoir):
    input_vec = torch.rand((1, deep_reservoir.input_size))
    out = deep_reservoir.forward(input_vec)
    assert out.shape == (1, 1000)


def test_reset_activations(deep_reservoir: DeepReservoir):
    deep_reservoir.reset_reservoir_state()