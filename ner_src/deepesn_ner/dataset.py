from typing import List, Tuple, Generator, Sequence
import logging
from torch import Tensor
import torch
from torch.utils.data import Dataset
from flair.data import Dictionary, Sentence


class FlatDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        if features.device != targets.device:
            raise TypeError(f"Device mismatch between feature and target tensors: {features.device} vs "
                            f"{targets.device}")
        self.features: torch.Tensor = features
        self.targets: torch.Tensor = targets

    @property
    def device(self):
        return self.features.device

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, inx):
        return self.features[inx], self.targets[inx]

    def to(self, device):
        if device == self.features.device:
            return self
        features = self.features.to(device)
        targets = self.targets.to(device)
        return FlatDataset(features, targets)


class EmbeddedNERDataset(Dataset):
    def __init__(self, sentences: List[Sentence], features: List[Tensor], targets: List[Tensor], tag_mapping: Dictionary,
                 split_name, bidirectional):
        """

        Args:
            sentences: a list of Flair sentences corresponding to the sentences in the next arguments
            features: a list of (n, D) tensors where n is the given sequence's lengths and D the embedding dim. n varies
            between samples
            targets: a list of (n) tensors with targets for the sequence's tokens where n is the sequence length which
            varies between samples
            tag_mapping: the mapping between NER tags and inx
            split_name: the name of the split
            bidirectional: if true this is a bidirectional reservoir and the first half of it's features dimensions
                correspond to the forward embeddings
        """
        super().__init__()
        self.split_name = split_name
        # Tests if all features are in the same device
        device = features[0].device
        for sample in features:
            if sample.device != device:
                raise TypeError(f"Mismatch in (device-)location of samples' features tensors:"
                                f" {device} vs {sample.device}")
        for target in targets:
            if target.device != device:
                raise TypeError("Mismatch in (device-)location of of target and feature tensors")
        self._device = device

        if len(features) != len(targets):
            raise ValueError(f"number of sentences provided in features is not the same as the number in the"
                             f" targets:"
                             f"{len(features)} vs {len(targets)}")

        for sentence_inx in range(len(features)):
            st_features = features[sentence_inx]
            st_targets = targets[sentence_inx]
            if st_features.shape[0] != st_targets.shape[0]:
                raise ValueError(f"number of tokens provided in features is not the same as the number in the"
                                 f" targets:"
                                 f"{st_features.shape} vs {st_targets.shape}")
            if bidirectional:
                if st_features.shape[1] % 2 != 0:
                    # checks if there is an even number of dimensions in the feature vector
                    raise ValueError(f"Features tensor of sentence with inx {sentence_inx} has an odd number of"
                                     f"embedding dimensions which is incompatible with a bidirectional dataset")

        self.features = features

        """List of tensors with the targets"""
        self.targets = targets

        self.sentences = sentences
        self.tag_mapping = tag_mapping

        self._bidirectional = bidirectional

        """defines if it returns bidirectional or unidirectional embeddings on getitem"""
        self._bidirectional_mode = bidirectional

    def bidirectional_mode(self):
        if self._bidirectional is False:
            raise TypeError("Cannot enable bidirectional mode with unidirectional embeddings")
        self._bidirectional_mode = True

    def unidirectional_mode(self):
        self._bidirectional_mode = False

    @property
    def in_bidirectional_mode(self):
        return self._bidirectional_mode

    @property
    def is_bidirectional(self):
        return self._bidirectional

    def to(self, device):
        """Creates a new dataset instance with data stored in another device. If `device` is already the current device
        returns the same instance"""
        logging.info(f"moving dataset {self.split_name} to {device}")
        if device == self._device:
            return self
        features = [ten.to(device) for ten in self.features]
        targets = [ten.to(device) for ten in self.targets]
        ds = EmbeddedNERDataset(self.sentences, features, targets, self.tag_mapping, self.split_name,
                                self._bidirectional)
        ds._bidirectional_mode = self._bidirectional_mode
        return ds

    @staticmethod
    def from_pickle(path):
        dataset: EmbeddedNERDataset = torch.load(path, map_location="cpu")
        dataset._device = "cpu"
        return dataset

    @property
    def device(self):
        return self._device

    @property
    def embedding_size(self) -> int:
        """ returns the embedding size. If the embeddings are bidirectional and the bidirectional mode is disabled,
        the size returned is only half of the size of the (bidirectional) embeddings stored
        """
        if self._bidirectional and not self._bidirectional_mode:
            return self.features[0].shape[1] // 2
        else:
            return self.features[0].shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, x) -> Tuple[Sentence, torch.Tensor, torch.Tensor]:
        features = self.features[x]
        # if x is an slice it features is a list
        if not isinstance(features, Sequence):
            features = [features]

        if self._bidirectional and self._bidirectional_mode is False:
            uni_size = self.embedding_size
            for inx, sample in enumerate(features):
                features[inx] = features[:, :uni_size]

        if len(features) == 1:
            features = features[0]

        return self.sentences[x], features, self.targets[x]

    def flatten(self):
        """
        Takes a dataset organized by sentence and flattens it into a token level dataset
        Args:
        Returns: a flattened dataset
        """
        # need to create a flattened dataset for GD
        features = torch.cat(self.features, dim=0)
        targets = torch.cat(self.targets, dim=0)
        return FlatDataset(features, targets)


class EfficientEmbeddedNERDataset(Dataset):
    """ Implements a dataset that stores embeddings of many sentences in a flat 2d tensor and stores indices for each
     sentence"""

    def __init__(self, sentences: List[Sentence], features: Tensor, targets: Tensor, sentence_index: Tensor,
                 tag_mapping: Dictionary, split_name, bidirectional):
        """
        Args:
            sentences: a list of Flair sentences corresponding to the sentences in the next arguments
            features: a (n, D) tensors where n is the number of tokens in all sentences and D the embedding dim.
            targets: a list of (n) tensors with targets for the sequence's tokens where n is the number of tokens in
            all sentences
            sentence_index: a (s) tensor which contains the starting index (in features, targets) of the tokens of
                the sentence at each location
            tag_mapping: the mapping between NER tags and inx
            split_name: the name of the split
            bidirectional: if true this is a bidirectional reservoir and the first half of it's features dimensions
                correspond to the forward embeddings
        """
        super().__init__()
        self.split_name = split_name
        # Tests if all features are in the same device
        if features.device != targets.device:
            raise TypeError(f"Device mismatch between feature and target tensors: {features.device} vs "
                            f"{targets.device}")
        self._device = features.device

        if features.shape[0] != targets.shape[0]:
            raise ValueError(f"number of tokens provided in features is not the same as the number in the"
                             f" targets:"
                             f"{len(features)} vs {len(targets)}")

        if len(sentences) != sentence_index.shape[0]:
            raise ValueError(f"number of sentences provided in sentences attribute is not the same as the number in the"
                             f" sentence_index:"
                             f"{len(sentences)} vs {len(sentence_index)}")

        self._features = features

        """List of tensors with the targets"""
        self.targets = targets
        self.sentence_index = sentence_index
        self.sentences = sentences
        self.tag_mapping = tag_mapping
        self._bidirectional = bidirectional

        """defines if it returns bidirectional or unidirectional embeddings on getitem"""
        self._bidirectional_mode = bidirectional

    def bidirectional_mode(self):
        if self._bidirectional is False:
            raise TypeError("Cannot enable bidirectional mode with unidirectional embeddings")
        self._bidirectional_mode = True

    def unidirectional_mode(self):
        self._bidirectional_mode = False

    def to(self, device):
        """Creates a new dataset instance with data stored in another device. If `device` is already the current device
        returns the same instance"""
        if device == self._device:
            return self
        features = self._features.to(device)
        targets = self.targets.to(device)
        ds = EfficientEmbeddedNERDataset(self.sentences, features, targets, self.sentence_index,
                                         self.tag_mapping, self.split_name, self._bidirectional)
        ds._bidirectional_mode = self._bidirectional_mode
        return ds

    @staticmethod
    def from_pickle(path):
        dataset: EfficientEmbeddedNERDataset = torch.load(path, map_location="cpu")
        dataset._device = "cpu"
        return dataset

    @property
    def device(self):
        return self._device

    @property
    def features(self) -> torch.Tensor:
        """a tensor with all features. If datasetis bidirectional in unidirectional mode, returns only forward embd"""
        embd_dim = self.embedding_size
        return self._features[:, :embd_dim]

    @property
    def embedding_size(self) -> int:
        """ returns the embedding size. If the embeddings are bidirectional and the bidirectional mode is disabled,
                the size returned is only half of the size of the (bidirectional) embeddings stored
        """
        if self._bidirectional and not self._bidirectional_mode:
            return self._features.shape[1] // 2
        else:
            return self._features.shape[1]

    def __len__(self):
        return self._features.shape[0]

    def __getitem__(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets tokens embedding at index or slice x. If dataset is bidirectional and unidirectional mode is enabled,
        returns only the forward embeddings"""

        if self._bidirectional and not self._bidirectional_mode:
            uni_size = self.embedding_size
            features = self._features[x, :uni_size]
        else:
            features = self._features[x]

        if type(x) == slice:
            return features, self.targets[x]

        if x > 0:
            prev_targets = self.targets[x-1]
        else:
            prev_targets = torch.zeros_like(self.targets[x])
        return features, self.targets[x], prev_targets

    def from_sentence(self, sentence_position):
        """Returns features and targets from sentence. If dataset is bidirectional and unidirectional mode is enabled,
        returns only the forward embeddings"""
        token_range_start = self.sentence_index[sentence_position]
        if sentence_position == len(self.sentences)-1:
            # its the last sentence
            token_range_end = len(self)
        else:
            token_range_end = self.sentence_index[sentence_position+1]

        return self[token_range_start:token_range_end][:2]

    def sentence_iterator(self) -> Generator[Tuple[Sentence, torch.Tensor, torch.Tensor], None, None]:
        for inx in range(len(self.sentences)):
            feat, target = self.from_sentence(inx)
            yield self.sentences[inx], feat, target

    @staticmethod
    def from_embedded_dataset(old: EmbeddedNERDataset):
        sentence_inx = []
        last_inx = 0
        for inx in range(len(old.sentences)):
            sentence_inx.append(last_inx)
            last_inx += old.features[inx].shape[0]

        flattened = old.flatten()
        sentence_inx = torch.tensor(sentence_inx)
        eds = EfficientEmbeddedNERDataset(old.sentences, flattened.features, flattened.targets, sentence_inx,
                                          old.tag_mapping, old.split_name, old.is_bidirectional)
        eds._bidirectional_mode = old.in_bidirectional_mode
        return eds

    def to_embedded_dataset(self) -> EmbeddedNERDataset:
        sentences = []
        features = []
        targets = []
        for sentence, feature_tensor, target_tensor in self.sentence_iterator():
            sentences.append(sentence)
            features.append(feature_tensor)
            targets.append(target_tensor)
        return EmbeddedNERDataset(sentences, features, targets, self.tag_mapping, self.split_name, self._bidirectional)

    @property
    def is_bidirectional(self):
        return self._bidirectional

    @property
    def in_bidirectional_mode(self):
        return self._bidirectional_mode
