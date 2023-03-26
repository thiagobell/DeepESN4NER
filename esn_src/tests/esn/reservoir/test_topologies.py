import pytest

import numpy as np

from esn_toolkit.reservoir.topologies import InputTopology, ReservoirTopology


@pytest.fixture
def reservoir_size():
    yield 100


@pytest.fixture
def connectivity():
    yield 0.8


@pytest.fixture
def scaling():
    yield 0.5


class TestReservoirTopology:
    @pytest.fixture
    def weight_matrix(self, reservoir_size, scaling, connectivity):
        weights = ReservoirTopology.create_random(reservoir_size, scaling, connectivity)
        yield weights

    def test_generated_non_fixed(self, weight_matrix, reservoir_size):
        assert weight_matrix.shape == (reservoir_size, reservoir_size)

    def test_scale(self, weight_matrix, scaling):
        assert weight_matrix.max() <= scaling
        assert weight_matrix.min() >= -scaling

    def test_connectivity(self, weight_matrix, reservoir_size, connectivity):
        max_edges = reservoir_size*(reservoir_size - 1)

        non_negative_map = weight_matrix != 0
        ratio = float(non_negative_map.sum()) / float(max_edges)
        assert abs(connectivity - ratio) < 0.05

    def test_generated_fixed(self, reservoir_size, scaling, connectivity):
        weights = ReservoirTopology.create_random(reservoir_size, scaling, connectivity, 1.2)
        assert weights.shape == (reservoir_size, reservoir_size)

        w_f = (weights == 1.2).numpy()
        w_0 = (weights == 0).numpy()
        # weights are either 1.2 (fixed value)  or 0 (no edge exists)
        assert np.all(np.logical_or(w_f, w_0))


class TestInputTopology:
    @pytest.fixture
    def input_size(self):
        yield 10

    @pytest.mark.parametrize("bias", [True, False])
    def test_generated(self, input_size, reservoir_size, scaling, connectivity, bias):
        weights = InputTopology.create_random(input_size, reservoir_size, bias, scaling, connectivity)
        if bias:
            input_dimensions = input_size+1
        else:
            input_dimensions = input_size
        assert weights.shape == (reservoir_size, input_dimensions)

