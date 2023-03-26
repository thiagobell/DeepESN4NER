"""
Implements the ESN applied to NER
"""
import re
import logging
import os
from typing import Union, List, Dict, Any

import torch
import sys
from torch.utils.data import DataLoader, WeightedRandomSampler
from functools import partial
import json
from pathos.multiprocessing import Pool
from pathos.helpers import mp
from torch.nn import CrossEntropyLoss, LogSoftmax
from collections import Counter
import time
from p_tqdm import p_map

from flair.data import Dictionary, Sentence, Label, Token
from flair.training_utils import Metric
from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score
from esn_toolkit.reservoir import DeepReservoir, Reservoir, Bi, BaseReservoir, Identity
from deepesn_ner.config import esn_embeddings_folder, ner_models_folder


from deepesn_ner.dataset import EmbeddedNERDataset, FlatDataset, EfficientEmbeddedNERDataset


class NerEsn(torch.nn.Module):
    def __init__(self, embedding_dimension, tag_mapping: Dictionary, recurrent_output=False, device="cpu"):
        """

        Args:
            embedding_dimension:
            tag_mapping:
            recurrent_output: if true, prediction of last token is given as input to output linear layer
            device:
        """
        super(NerEsn, self).__init__()
        self.output_dimension = len(tag_mapping)
        self.embedding_dimension = embedding_dimension
        self.tag_mapping = tag_mapping
        self.device = device
        self.recurrent_output = recurrent_output
        self.read_out_layer = torch.nn.Linear(embedding_dimension, self.output_dimension).to(device)

    def to(self, device):
        """returns a new instance of the model with the given device """
        read_out = self.read_out_layer.to(device)
        n_instance = NerEsn(self.embedding_dimension, self.tag_mapping, recurrent_output=self.recurrent_output,
                            device=device)
        n_instance.read_out_layer = read_out
        return n_instance

    def train_model(self, train_dataset: EfficientEmbeddedNERDataset, dev_dataset: EfficientEmbeddedNERDataset, num_epochs,
                    opt: str = "adam",
                    lr=1e-3, decay=1e-5,
                    batch_size=64, balance_dataset=False):
        # to train this dataset we have to first flatten it
        print(f"self.device is {self.device}")
        if self.recurrent_output:
            self.read_out_layer = torch.nn.Linear(self.embedding_dimension+self.output_dimension,
                                                  self.output_dimension).to(self.device)
        else:
            self.read_out_layer = torch.nn.Linear(self.embedding_dimension,
                                                  self.output_dimension).to(self.device)
        print("dataloader")
        # maps classes to its number of samples
        if balance_dataset:
            print("sampling dataset so as to balance it across labels")
            target_list = train_dataset.targets.tolist()
            class_count = Counter(target_list)
            num_samples = len(train_dataset.targets)
            print(f"there are {num_samples} samples")
            weights = [num_samples/class_count[c] for c in target_list]
            sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
            training_data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        else:
            training_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if opt == "adadelta":
            optimizer = torch.optim.Adadelta(self.read_out_layer.parameters(), lr=lr, weight_decay=decay)
        elif opt == "adam":
            optimizer = torch.optim.Adam(self.read_out_layer.parameters(), lr=lr, weight_decay=decay)
        elif opt == "sgd":
            optimizer = torch.optim.SGD(self.read_out_layer.parameters(), lr=lr, weight_decay=decay)
        else:
            print(f"invalid opt: {opt}")
            return f"invalid opt: {opt}"
        loss_fn = CrossEntropyLoss()
        loss_values = []
        report = []
        sys.stdout.flush()

        for epoch in range(num_epochs):
            losses = 0.0
            count = 0
            for features, targets, prev_target in training_data_loader:
                optimizer.zero_grad()

                if self.recurrent_output:
                    prev_target_one_hot = torch.zeros((targets.shape[0], self.output_dimension), device=features.device)
                    prev_target_one_hot[range(targets.shape[0]), prev_target] = 1
                    input_read_out = torch.cat([prev_target_one_hot, features], dim=1)
                else:
                    input_read_out = features

                output = self.read_out_layer.forward(input_read_out)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                losses += loss.item() * features.shape[0]
                count += features.shape[0]
            losses = losses/count
            loss_values.append(losses)
            del output
            epoch_report = dict()
            epoch_report["train_loss"] = losses
            if epoch % 5 == 4:
                entity_metrics_train = self.entity_eval(train_dataset, print_report=False)
                entity_metrics_dev = self.entity_eval(dev_dataset, print_report=True)
                token_metrics_train = self.token_eval(train_dataset)
                token_metrics_dev = self.token_eval(dev_dataset)
                print(f"Epoch {epoch+1}/{num_epochs}, Training Loss {losses:.4f}, Entity(micro/macro) "
                      f"{entity_metrics_train['micro f1']:.2f}/{entity_metrics_train['macro f1']:.2f}, "
                      f"Token(micro/macro) {token_metrics_train['classification report']['micro avg']['f1-score']:.2f}/"
                      f"{token_metrics_train['classification report']['macro avg']['f1-score']:.2f}; "
                      f"dev Entity(micro/macro) {entity_metrics_dev['micro f1']:.2f}/"
                      f"{entity_metrics_dev['macro f1']:.2f}"
                      f"Token(micro/macro) {token_metrics_dev['classification report']['micro avg']['f1-score']:.2f}/"
                      f"{token_metrics_dev['classification report']['macro avg']['f1-score']:.2f}; ")

                epoch_report["entity_train"] = entity_metrics_train
                epoch_report["entity_dev"] = entity_metrics_dev

                epoch_report["token_train"] = token_metrics_train
                epoch_report["token_dev"] = token_metrics_dev
                torch.cuda.empty_cache()

            else:
                print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss {losses:.4f}")

            report.append(epoch_report)

        return report

    def token_eval(self, embedded_dataset: EfficientEmbeddedNERDataset, print_report=False):
        """
        Evaluate predictions on the token level
        Args:
            embedded_dataset:
            print_report:

        Returns:

        """
        loss_fn = CrossEntropyLoss()
        softmax_layer = LogSoftmax(1)
        with torch.no_grad():
            if self.recurrent_output:
                # TODO PREV PREDICTION IS A DUMMY!!!!
                print("TOKEN EVAL PREDICTION IS A DUMMY!!!!")
                prev_pred = torch.zeros((embedded_dataset.features.shape[0], self.output_dimension), device=embedded_dataset.features.device)
                unnormalized_probs = self.read_out_layer.forward(torch.cat([prev_pred, embedded_dataset.features], dim=1))
            else:
                unnormalized_probs = self.read_out_layer.forward(embedded_dataset.features)
            loss = loss_fn(unnormalized_probs, embedded_dataset.targets)
            predictions_probs = softmax_layer(unnormalized_probs)
            predictions_tag_inx = predictions_probs.argmax(dim=1)

            # we need to make sure the predictions are in the cpu
            predictions_tag_inx = predictions_tag_inx.cpu()
            # returns sorted list of tags
            target_names = self.tag_mapping.get_items()
            targets_cpu = embedded_dataset.targets.cpu()
            if print_report:
                report = classification_report(targets_cpu, predictions_tag_inx,
                                               labels=list(range(len(self.tag_mapping))), target_names=target_names)
                print(report)

            dict_report = classification_report(targets_cpu, predictions_tag_inx,
                                                labels=list(range(len(self.tag_mapping))), target_names=target_names,
                                                output_dict=True)

            return {"loss": loss.item(), "classification report": dict_report}

    def entity_eval(self, embedded_dataset: Union[EmbeddedNERDataset, EfficientEmbeddedNERDataset], print_report = False):
        """
        Evaluates the predictions at the entity level.
        Based off of https://github.com/flairNLP/flair/blob/master/flair/models/sequence_tagger_model.py
        Args:
            embedded_dataset: the dataset with embeddings

        Returns:
        """
        softmax_layer = LogSoftmax(1)

        metric = Metric("entity eval")

        if isinstance(embedded_dataset, EmbeddedNERDataset):
            sentence_iterator = embedded_dataset
        elif isinstance(embedded_dataset, EfficientEmbeddedNERDataset):
            sentence_iterator = embedded_dataset.sentence_iterator()
        else:
            raise TypeError(f"{type(embedded_dataset)} is not a supported dataset type")

        with torch.no_grad():
            for sentence, sentence_embd, _ in sentence_iterator:
                sentences_predictions = []
                predictions_tag_inx = 0
                for token_inx in range(len(sentence)):
                    if self.recurrent_output:
                        prev_output = torch.zeros((1, self.output_dimension), device=sentence_embd.device)
                        prev_output[0, predictions_tag_inx] = 1
                        unnormalized_probs = self.read_out_layer.forward(torch.cat([prev_output, sentence_embd[[token_inx]]], dim=1))
                    else:
                        unnormalized_probs = self.read_out_layer.forward(sentence_embd[[token_inx]])
                    predictions_probs = softmax_layer(unnormalized_probs)
                    predictions_tag_inx: torch.Tensor = predictions_probs.argmax(dim=1)
                    predictions_tag_inx = predictions_tag_inx.cpu()[0]
                    sentences_predictions.append(predictions_tag_inx)

                predicted_tags = [self.tag_mapping.get_item_for_index(code.item()) for code in sentences_predictions]

                # add predictions to tokens
                for token, tag in zip(sentence, predicted_tags):
                    token.add_tag_label("prediction_ner", Label(tag))

                # get predicted spans
                # we use the repr of each span to compare them effectively since the Span class does not implement
                # __cmp__
                if type(sentence) == Token:
                    print(f"sentence {sentence} is token")
                predicted_spans = sentence.get_spans("prediction_ner")
                predicted_span_repr = [repr(sp) for sp in predicted_spans]
                target_spans = sentence.get_spans("ner")
                target_span_repr = [repr(sp) for sp in target_spans]

                # calculate metrics
                for predicted_rep, predicted_span in zip(predicted_span_repr, predicted_spans):
                    if predicted_rep in target_span_repr:
                        # we have a correct classification of a span
                        metric.add_tp(predicted_span.tag)
                    else:
                        # we predicted a span incorrectly
                        metric.add_fp(predicted_span.tag)

                # we check for spans in the target that were not present
                for target_rep, target_span in zip(target_span_repr, target_spans):
                    if target_rep not in predicted_span_repr:
                        # not found
                        metric.add_fn(target_span.tag)

        if print_report:
            print(metric)
        micro_f1 = metric.micro_avg_f_score()
        macro_f1 = metric.macro_avg_f_score()
        micro_acc = metric.micro_avg_accuracy()
        macro_acc = metric.macro_avg_accuracy()

        return {"micro f1": micro_f1, "macro f1": macro_f1, "micro acc": micro_acc, "macro acc": macro_acc,
                "classification report": str(metric)}


def get_robins_rs():
    rs = Bi(Reservoir(300, True, reservoir_size=2048, input_weight_scale=0.25, input_connectivity=0.5,
                      spectral_radius=0.7,
                      reservoir_connectivity=0.5, reservoir_weight_scale=0.5,
                      leaking_rate=0.75, available_activation_functions=["tanh"]))
    return rs


def get_example_rs():
    rs = Bi(Reservoir(300, enable_bias=True, reservoir_size=500, input_weight_scale=0.5, input_connectivity=0.5,
                      spectral_radius=0.79,
                      reservoir_connectivity=0.5, reservoir_weight_scale=0.5,
                      leaking_rate=0.3, available_activation_functions=[torch.tanh]))
    return rs


def train_esn(esn_name, model_name, train_parameters: Dict[str, Any], bidirectional, device, esn_trial_number=None):
    """
    Trains with pre-generated embeddings in embeddings folder
    Returns:

    Args:
        esn_name:
        model_name:
        train_parameters:
        bidirectional:
            if dataset is bidirectional AND:
                * bidirectional = true -> trains model with a bidirectional embedding
                * bidirectional = False -> trains model with only FORWARD embeddings
            if dataset is unidirectional AND:
                * bidirectional = true -> ERROR
                * bidirectional = False -> trains model with the unidirectional embeddings provided
        device: the device to train the network with
        esn_trial_number: the trial number of the esn embeddings being used as features

    Returns:

    """
    embd_folder = os.path.join(esn_embeddings_folder, esn_name)
    model_folder = os.path.join(ner_models_folder, esn_name + "--" + model_name)
    print("hi")

    if esn_trial_number is not None:
        trial_str = f".{esn_trial_number}"
    else:
        trial_str = ""

    if os.path.exists(os.path.join(model_folder, "train_report"+trial_str+".json")):
        print("model already trained")
        return

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # save parameters
    with open(os.path.join(model_folder, "train.param"), "w") as f:
        json.dump(train_parameters, f)

    print("loaded res")
    train_ds: EfficientEmbeddedNERDataset = EfficientEmbeddedNERDataset.from_pickle(os.path.join(embd_folder,
                                                                                                 "train"+trial_str+".pickle"))
    train_ds = train_ds.to(device)
    print("loaded train ds")
    dev_ds: EfficientEmbeddedNERDataset = EfficientEmbeddedNERDataset.from_pickle(os.path.join(embd_folder,
                                                                                               "dev"+trial_str+".pickle"))
    dev_ds = dev_ds.to(device)
    print("loaded dev ds")
    if train_ds.is_bidirectional != dev_ds.is_bidirectional:
        format_ds = lambda ds: f"{'bidirectional' if ds.is_bidirectional else 'unidirectional'}"
        raise ValueError(f"Datasets dont match: train dataset is {format_ds(train_ds)} while dev ds is"
                         f" {format_ds(dev_ds)}")

    # VERY important: This configures the behaviour of the dataset
    if bidirectional:
        if train_ds.is_bidirectional:
            train_ds.bidirectional_mode()
            dev_ds.bidirectional_mode()
        else:
            raise ValueError("Cannot use a unidirectional dataset for bidirectional training")
    else:
        train_ds.unidirectional_mode()
        dev_ds.unidirectional_mode()

    print("init ner class")
    print(f"train_ds tag mappint:type:{type(train_ds.tag_mapping)}")
    recurrent_output = train_parameters.get("recurrent_output", False)
    ner = NerEsn(train_ds.embedding_size, train_ds.tag_mapping, recurrent_output=recurrent_output, device=device)

    print("training")

    # we need to remove the recurrent_output parameter if it is set
    if "recurrent_output" in train_parameters:
        del train_parameters["recurrent_output"]
    train_report = ner.train_model(train_ds, dev_ds, **train_parameters)

    with open(os.path.join(model_folder, "train_report"+trial_str+".json"), "w") as f:
        json.dump(train_report, f)

    # saving ner
    torch.save(ner, os.path.join(model_folder, "model"+trial_str+".pickle"))


def parallel_train(esn_name, model_name, train_parameters: Dict[str, Any], bidirectional,
              device_access_control, device_access_lock):
    """
    Trains with pre-generated embeddings in embeddings folder
    Returns:

    Args:
        esn_name:
        model_name:
        train_parameters:
        bidirectional:
            if dataset is bidirectional AND:
                * bidirectional = true -> trains model with a bidirectional embedding
                * bidirectional = False -> trains model with only FORWARD embeddings
            if dataset is unidirectional AND:
                * bidirectional = true -> ERROR
                * bidirectional = False -> trains model with the unidirectional embeddings provided
        device_access_control:
        device_access_lock:

    Returns:

    """
    # gets a lock on a gpu
    if device_access_control is None:
        device_inx = -1
        device = "cpu"
    else:
        device_inx = None
        device = None

    while device_inx is None:
        try:
            print("getting lock")
            with device_access_lock:
                # get the gpu with the maximum number of free slots
                max_avail_slots = max(device_access_control)
                if max_avail_slots == 0:
                    # no free slots
                    raise ValueError
                device_inx = device_access_control.index(max_avail_slots)

                device_access_control[device_inx] -= 1
                print(f"acquired lock on device inx: {device_inx}")
            print(device_access_control)
            device = f"cuda:{device_inx}"
        except ValueError:
            # no device is free
            print("could not find any free device. waiting 60seconds")
            time.sleep(60)
    embd_folder = os.path.join(esn_embeddings_folder, esn_name)
    # checks if folder exists:
    if not os.path.exists(embd_folder):
        # wish i could raise an exception here but multiprocessing seems to ignore them
        print(f"no embedding folder found for {embd_name}")

    esn_folder_contents = os.listdir(embd_folder)
    if "train.pickle" in esn_folder_contents:
        # there is a single trial in this folder and there is no numbering
        num_trials = None
        print(f"1 trial found for esn: {esn_name}")
    else:
        trial_parser = r"train.([0-9]*).pickle"
        num_trials = 0
        for file_name in esn_folder_contents:
            trial_ls = re.findall(trial_parser, file_name)
            if len(trial_ls) > 0:
                num_trials = max(num_trials, int(trial_ls[0])+1)
        print(f"{num_trials} trials found for esn: {esn_name}")
        if num_trials == 0:
            print(f"0 trials found for {embd_name}")
    try:
        model_folder = os.path.join(ner_models_folder, esn_name + "--" + model_name)
        if num_trials is None:

            # checks if the model hasnt already been trained
            if os.path.exists(os.path.join(model_folder, "train_report.json")):
                print(f"single trial of model {model_name} already trained")
                return
            train_esn(esn_name, model_name, train_parameters, bidirectional, device, None)
        else:
            #
            for trial in range(num_trials):
                if os.path.exists(os.path.join(model_folder, f"train_report.{trial}.json")):
                    print(f"trial {trial} of model {model_name} has already been trained")
                    return
                print(f"training trial {trial}")
                train_esn(esn_name, model_name, train_parameters, bidirectional, device, trial)
                torch.cuda.empty_cache()

    except Exception as e:
        logging.critical(f"{esn_name+model_name}: Exception raised: {e}")
        print(f"{esn_name + model_name}: Exception raised: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if device_access_control is not None:
            # free device
            with device_access_lock:
                device_access_control[device_inx] += 1
            print(f"releasing lock on device inx {device_inx}")
            print(device_access_control)
            torch.cuda.empty_cache()
    print("exiting parallel process")

def check_dataset(esn_name):
    embd_folder = os.path.join(esn_embeddings_folder, esn_name)
    print(f"checking {esn_name}")
    pickle_names = ["train.pickle", "test.pickle", "dev.pickle"]
    for split in pickle_names:
        fpath = os.path.join(embd_folder, split)
        ds = EmbeddedNERDataset.from_pickle(fpath)
        if isinstance(ds, EmbeddedNERDataset):
            print(f"{esn_name}/{split} is in old format converting....", end="")
            ds = EfficientEmbeddedNERDataset.from_embedded_dataset(ds)
            torch.save(ds, fpath)
            print("done")
        elif isinstance(ds, EfficientEmbeddedNERDataset):
            changed = False
            if "_features" not in ds.__dict__:
                ds._features = ds.__dict__["features"]
                ds._bidirectional = True
                ds._bidirectional_mode = True
                changed = True

            if "_bidirectional" not in ds.__dict__:
                ds._bidirectional = False
                ds._bidirectional_mode = False
                changed = True
            if changed:
                torch.save(ds, fpath)
if __name__ == "__main__":
    #bi_flair_identity()

    #word_embd_biesn()
    #deep_exp()

    """esn_embeddings = [
        "uni-esn1L-flair+word", "bi-esn1L-flair+word", "uni-esn2L-flair+word", "bi-esn2L-flair+word",
        "uni-esn3L-flair+word", "bi-esn3L-flair+word", "uni-esn4L-flair+word", "bi-esn4L-flair+word"
    ]
    """
    """
    esn_embeddings = [
        "word_gs_2L_bi_:0.95-0.95",
        "word_gs_2L_bi_:0.95-0.75",
        "word_gs_2L_bi_:0.95-0.55",
        "word_gs_2L_bi_:0.75-0.95",
        "word_gs_2L_bi_:0.75-0.75",
        "word_gs_2L_bi_:0.75-0.55",
        "word_gs_2L_bi_:0.55-0.95",
        "word_gs_2L_bi_:0.55-0.75",
        "word_gs_2L_bi_:0.55-0.55",
    ]
    """
    esn_embeddings = [
        #"word_gs_2L_bi_:0.55-0.95",
        #"word_gs_2L_bi_:0.55-0.75",
        #"word_gs_2L_bi_:0.55-0.55",
        #"word_gs_2L_bi_:spec_radius0.5-0.5",
        #"word_gs_2L_bi_:spec_radius0.5-0.7",
        #"word_gs_2L_bi_:spec_radius0.5-0.99",
        #"word_gs_2L_bi_:spec_radius0.7-0.5",
        #"word_gs_2L_bi_:spec_radius0.7-0.7",
        #"word_gs_2L_bi_:spec_radius0.7-0.99",
        #"word_gs_2L_bi_:spec_radius0.99-0.5",
        #"word_gs_2L_bi_:spec_radius0.99-0.7",
        #"word_gs_2L_bi_:spec_radius0.99-0.99"
        #"conll2003_word_GS_1024-1024_rc0.1_lk0.9_sr0.7_ic0.1",
        #"conll2003_word_GS_1024-1024_rc0.1_lk0.9_sr0.7_ic0.5",
        #"conll2003_word_GS_1024-1024_rc0.1_lk0.9_sr0.9_ic0.1",
        #"conll2003_word_GS_1024-1024_rc0.1_lk0.9_sr0.9_ic0.5",
        #"conll2003_word_GS_1024-1024_rc0.1_lk0.7_sr0.7_ic0.1",
        #"conll2003_word_GS_1024-1024_rc0.1_lk0.7_sr0.7_ic0.5",
        #"conll2003_word_GS_1024-1024_rc0.1_lk0.7_sr0.9_ic0.1",
        #"conll2003_word_GS_1024-1024_rc0.1_lk0.7_sr0.9_ic0.5"
        #"conll2003_word_GS_2048_rc0.1_lk0.7_sr0.7_ic0.1",
        #"conll2003_word_GS_2048_rc0.1_lk0.7_sr0.7_ic0.5",
        #"conll2003_word_GS_2048_rc0.1_lk0.7_sr0.9_ic0.1",
        #"conll2003_word_GS_2048_rc0.1_lk0.7_sr0.9_ic0.5",
        #"conll2003_word_GS_2048_rc0.1_lk0.9_sr0.7_ic0.1",
        #"conll2003_word_GS_2048_rc0.1_lk0.9_sr0.7_ic0.5",
        #"conll2003_word_GS_2048_rc0.1_lk0.9_sr0.9_ic0.1",
        #"conll2003_word_GS_2048_rc0.1_lk0.9_sr0.9_ic0.5"
        #"conll2003_word_GS_1024-512-512_rc0.1_lk0.7_sr0.7_ic0.5",
        "conll2003_word_GS_1024-512-512_permutated_lk0.7_sr0.7_ic0.5",
        "conll2003_word_GS_1024-512-512_permutated_lk0.7_sr0.9_ic0.5",
        "conll2003_word_GS_1024-1024_permutated_lk0.7_sr0.7_ic0.5",
        "conll2003_word_GS_1024-1024_permutated_lk0.7_sr0.9_ic0.5",
        "conll2003_word_GS_2048_permutated_lk0.7_sr0.7_ic0.5",
        "conll2003_word_GS_2048_permutated_lk0.7_sr0.9_ic0.5"
    ]
    """
    train_esn("word_gs_2L_bi_:spec_radius0.99-0.99", "output-feedback-lr1e-3decay1e-5", {"opt": "adadelta", "num_epochs": 200, "lr": 1e-3, "decay": 1e-5,
                                     "batch_size": 64}, True, "cuda:2", esn_trial_number = 0)
    train_esn("word_gs_2L_bi_:spec_radius0.99-0.99", "output-feedback-lr1e-3decay1e-6",
              {"opt": "adadelta", "num_epochs": 200, "lr": 1e-3, "decay": 1e-6,
               "batch_size": 64}, True, "cuda:2", esn_trial_number=0)
    train_esn("word_gs_2L_bi_:spec_radius0.99-0.99", "output-feedback-lr1e-1decay1e-6",
              {"opt": "adadelta", "num_epochs": 200, "lr": 1e-1, "decay": 1e-6,
               "batch_size": 64}, True, "cuda:2", esn_trial_number=0)
    """

    slot_per_gpu = 2
    num_gpu = 6
    devices = [f"cuda:{inx}" for inx in range(num_gpu)]
    mg = mp.Manager()
    device_access = mg.list([slot_per_gpu]*len(devices))
    device_access_lock = mg.Lock()

    #parallel_train("ud_english_word_GS_1024-1024_rc0.1_lk0.9_sr0.7_ic0.5", "TESTE", {"recurrent_output": True, "opt": "adadelta", "num_epochs": 301, "lr": 1e-3, "decay": 1e-5,
    #                                         "batch_size": 64, "balance_dataset": False}, True, device_access, device_access_lock)
    #exit()
    #
    #p = mp.get_context("spawn").Pool(slot_per_gpu*num_gpu)
    p = mp.get_context("spawn").Pool(slot_per_gpu*num_gpu)
    for balance_dataset in [False]:
        for bidirectional in [True]:
            for recurrent_output in [True, False]:
                for inx, embd_name in enumerate(esn_embeddings):
                    for batch_size in [64]:
                        base_name = f"{'bi-' if bidirectional else 'uni-'}-recurrent_out:{recurrent_output}" \
                                    f"-bs{batch_size}-" \
                                    f"{'balanced' if balance_dataset else 'unbalanced'}-"
                        p.apply_async(parallel_train,
                                      args=(embd_name,
                                            f"{base_name} 300 epochs adadelta. bs{batch_size} lr 1e-3 decay1e-5",
                                            {"recurrent_output": recurrent_output, "opt": "adadelta", "num_epochs": 301, "lr": 1e-3, "decay": 1e-5,
                                             "batch_size": batch_size, "balance_dataset": balance_dataset}, bidirectional, device_access, device_access_lock)
                                      )

                        """
                        p.apply_async(parallel_train,
                                      args=(embd_name,
                                            f"{base_name}  300 epochs adadelta. bs{batch_size} lr 1e-3 decay1e-6",
                                            {"recurrent_output": recurrent_output, "opt": "adadelta", "num_epochs": 301, "lr": 1e-3, "decay": 1e-6,
                                             "batch_size": batch_size, "balance_dataset": balance_dataset}, bidirectional, device_access, device_access_lock)
                                      )
                        """
                        p.apply_async(parallel_train,
                                      args=(embd_name,
                                            f"{base_name}  300 epochs adadelta. bs{batch_size} lr 1e-1 decay1e-5",
                                            {"recurrent_output": recurrent_output, "opt": "adadelta", "num_epochs": 301, "lr": 1e-1, "decay": 1e-5,
                                             "batch_size": batch_size, "balance_dataset": balance_dataset}, bidirectional, device_access, device_access_lock)
                                      )
                        """
                        p.apply_async(parallel_train,
                                      args=(embd_name,
                                            f"{base_name}  300 epochs adadelta. bs{batch_size} lr 1e-1 decay1e-6",
                                            {"recurrent_output": recurrent_output, "opt": "adadelta", "num_epochs": 301, "lr": 1e-1, "decay": 1e-6,
                                             "batch_size": batch_size, "balance_dataset": balance_dataset}, bidirectional, device_access, device_access_lock)
                                      )
                        
                        p.apply_async(parallel_train,
                                      args=(embd_name,
                                            f"{base_name} 300 epochs adadelta. bs{batch_size} lr 1e-3",
                                            {"recurrent_output": recurrent_output, "opt": "adadelta", "num_epochs": 301, "lr": 1e-3, "decay": 1e-5,
                                             "batch_size": batch_size, "balance_dataset": balance_dataset}, bidirectional, device_access, device_access_lock)
                                      )
                        
                        p.apply_async(parallel_train,
                                      args=(embd_name, f"{base_name} 300 epochs adam. bs{batch_size} lr 1e-3",
                                            {"recurrent_output": recurrent_output, "opt": "adam", "num_epochs": 301, "lr": 1e-3, "decay": 1e-5"batch_size": batch_size, "balance_dataset": balance_dataset},
                                            bidirectional, device_access, device_access_lock)
                                      )
                        
                        p.apply_async(parallel_train,
                                      args=(embd_name, f"{base_name} 300 epochs adadelta. bs{batch_size} lr 1e-1",
                                            {"recurrent_output": recurrent_output, "opt": "adadelta", "num_epochs": 301, "lr": 1e-1, "decay": 1e-5,
                                             "batch_size": batch_size, "balance_dataset": balance_dataset}, bidirectional, device_access, device_access_lock)
                                      )
                        
                        p.apply_async(parallel_train,
                                      args=(embd_name,
                                            f"{base_name} 300 epochs SGD. bs{batch_size} lr 1e-3",
                                            {"recurrent_output": recurrent_output, "opt": "sgd", "num_epochs": 301, "lr": 1e-3, "decay": 1e-5,
                                             "batch_size": batch_size, "balance_dataset": balance_dataset}, bidirectional, device_access, device_access_lock)
                                      )
                        
                        p.apply_async(parallel_train,
                                      args=(embd_name,
                                            f"{base_name} 300 epochs SGD. bs{batch_size} lr 1e-1",
                                            {"recurrent_output": recurrent_output, "opt": "sgd", "num_epochs": 301, "lr": 1e-1, "decay": 1e-5,
                                             "batch_size": batch_size, "balance_dataset": balance_dataset}, bidirectional, device_access, device_access_lock)
                                      )
                        """
    p.close()
    p.join()
    print("done")