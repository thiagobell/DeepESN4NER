from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import MSELoss

from esn_toolkit.reservoir.standard import Reservoir


class EmbeddingDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: Optional[torch.Tensor]):
        self.inputs = inputs
        self.targets = targets
        if self.targets is not None:
            if inputs.shape[0] != targets.shape[0]:
                raise ValueError(f"mismatch in number of samples between inputs ({inputs.shape}) and"
                                 f" targets ({targets.shape})")

    def __getitem__(self, item):
        if self.targets is None:
            return self.inputs[item]
        else:
            return self.inputs[item], self.targets[item]

    def __len__(self):
        return self.inputs.shape[0]


class TimeSeriesESN():
    def __init__(self, reservoir: Reservoir, output_dimension: int):
        super().__init__()
        self.reservoir = reservoir
        self.read_out_layer = torch.nn.Linear(self.reservoir.reservoir_size, output_dimension)

    def _make_training_dataset(self, dataset: torch.tensor, initial_transient: int) -> EmbeddingDataset:
        """
        Generates ESN embeddings from the training set to be used for training
        Args:
            dataset: a (n x input dimensions) tensor
            initial_transient: the time steps to ignore from training
        Returns: an EmbeddingDataset with  (n-1 x reservoir_size) activation matrix as inputs and a
         (n-1 x input_dimension) matrix as targets
        """
        num_samples = dataset.shape[0]

        if initial_transient > num_samples:
            raise ValueError(f"initial transient of {initial_transient} is larger than the number of samples")

        # apply input data to the reservoir
        self.reservoir.reset_reservoir_state()
        activations = []
        for sample_inx in range(num_samples-1):
            sample = dataset[sample_inx]
            sample = sample.reshape(1, -1)
            act = self.reservoir.forward(sample)
            if sample_inx >= initial_transient:
                # only add activations after the initial transient has been passed
                activations.append(act)

        activations = torch.cat(activations)
        targets = dataset[initial_transient+1:]
        return EmbeddingDataset(activations, targets)

    def train_readout(self, num_epochs: int, train_data: torch.tensor, initial_transient: int, weight_decay=0.005,
                      lr=1e-4):
        """
        Trains the readout layer given training data (in the form of reservoir embeddings)
        Args:
            num_epochs: number of epochs to train for
            train_data: a (n x d) matrix where n is the number of samples and d the number of dimensions of samples
                (the size of the reservoir)

        Returns:

        """
        optimizer = torch.optim.Adam(self.read_out_layer.parameters(), weight_decay=weight_decay, lr=lr) # weight_decay=0.1, lr=0.005
        loss_fn = MSELoss()

        dataset = DataLoader(self._make_training_dataset(train_data, initial_transient), batch_size=64, shuffle=True)

        for epoch in range(num_epochs):
            losses = 0.0
            for batch in dataset:
                inputs, targets = batch
                optimizer.zero_grad()
                output = self.read_out_layer.forward(inputs)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                losses += loss.item()*inputs.shape[0]
            # print(f"epoch{epoch}, loss{losses/len(train_data):e}")
        # print(f"len test_data {len(train_data)}")

    def predict(self, initial_input, horizon):
        """
        Predicts a time series for horizon steps
        Args:
            initial_input: an (1 x input dimension) matrix
            horizon:

        Returns: an (horizon x output dimension) matrix

        """
        predictions_ls = []
        activations_ls = []
        input_value = initial_input
        with torch.no_grad():
            while len(predictions_ls) < horizon:
                activation = self.reservoir.forward(input_value)
                activations_ls.append(activation)
                input_value = self.read_out_layer.forward(activation)
                predictions_ls.append(input_value)
        return torch.cat(predictions_ls), torch.cat(activations_ls)

    def warmup(self, inputs):
        """
        Warmsup the reservoir with the provided in puts
        Args:
            inputs: a (n x d) matrix of inputs where n is the number of samples and d the input dimension

        Returns:

        """
        for i_inx in range(inputs.shape[0]):
            self.reservoir.forward(inputs[i_inx,:].view(1, -1))

    def evaluate_recursive(self, test_data: torch.Tensor):
        """
        Evaluates the ESN recursively. Takes predicted output at time step t as input
        Args:
            test_data:  an (n x input dimension) matrix of inputs. its first row must contain the last sample
                of the train set to work as a "seed" so the reservoir can continue at the input time step it stopped
                (training_set_size -1)
            initial_transient: the number of timesteps to ignore

        Returns:

        """
        loss_fn = MSELoss()
        # the last value of the training set
        predictions, activations = self.predict(test_data[0, :].reshape((1, -1)), test_data.shape[0] - 1)
        loss = loss_fn(predictions, test_data[1:, :]).item()

        return predictions, activations, loss
