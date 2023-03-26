import os
from typing import Dict, Any
import json
from deepesn_ner.ner_esn import NerEsn
from deepesn_ner.dataset import EfficientEmbeddedNERDataset, EmbeddedNERDataset
from pytorch_optimize.optimizer import ESOptimizer
from pytorch_optimize.model import Model, SamplingStrategy
from pytorch_optimize.objective import Objective, Samples
import torch
from dataclasses import dataclass
from deepesn_ner.config import esn_embeddings_folder, ner_models_folder
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(init=True)
class BatchSamples(Samples):
    sentences: torch.Tensor
    inputs: torch.Tensor
    targets: torch.Tensor


class Accuracy(Objective):
    def __call__(self, model: Model, samples: EmbeddedNERDataset, device: str):
        esn_net: NerEsn = model.net
        samples = samples.to(device)
        report = esn_net.entity_eval(samples, False)
        return [report["micro f1"]]


def train_esn(esn_name, model_name, train_parameters: Dict[str, Any], bidirectional, device):
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

    Returns:

    """
    embd_folder = os.path.join(esn_embeddings_folder, esn_name)
    model_folder = os.path.join(ner_models_folder, esn_name + "--" + model_name)
    if os.path.exists(os.path.join(model_folder, "train_report.json")):
        print("model already trained")
        #return

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # save parameters
    with open(os.path.join(model_folder, "train.0.param"), "w") as f:
        json.dump(train_parameters, f)

    print("loaded res")
    train_ds: EmbeddedNERDataset = EfficientEmbeddedNERDataset.from_pickle(os.path.join(embd_folder, "test.0.pickle")).to_embedded_dataset()
    train_ds = train_ds.to(device)
    print("loaded train ds")
    dev_ds: EfficientEmbeddedNERDataset = EfficientEmbeddedNERDataset.from_pickle(os.path.join(embd_folder, "dev.0.pickle"))
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
    ner = NerEsn(train_ds.embedding_size, train_ds.tag_mapping, device)

    wrapped_ner = Model(ner, SamplingStrategy.ALL)

    #train_report = ner.train_model(train_ds, dev_ds, **train_parameters)
    # objective function (loss function)
    obj_measure = Accuracy()
    #devices=[f"cuda:{inx}" for inx in range(6)]
    devices =["cpu"]
    # optimizer
    es_optimizer = ESOptimizer(model=wrapped_ner, sgd_optimizer=torch.optim.Adadelta(ner.parameters(), lr=1e-3),
                               objective_fn=obj_measure, obj_weights=[1.0], sigma=1e-3, n_samples=100,
                               devices=devices, n_workers=6)

    dataloader = torch.utils.data.DataLoader(train_ds, 300, shuffle=True, collate_fn=lambda xs: (
        [x[0] for x in xs],
        [x[1] for x in xs],
        [x[2] for x in xs]
    ))

    reports = []

    for epoch in range(100):
        show_every = 1
        for i, (sentences, inputs, targets) in enumerate(tqdm(dataloader)):
            samples = EmbeddedNERDataset(sentences, inputs, targets, train_ds.tag_mapping, train_ds.split_name,
                                         train_ds.is_bidirectional)
            es_optimizer.gradient_step(samples)
        if (epoch + 1) % show_every == 0:
            report = ner.entity_eval(dev_ds, False)
            print(report)
            reports.append(report)

    with open("raj_opt_reports.json", "w") as f:
        json.dump(reports, f)


    #with open(os.path.join(model_folder, "train_report.json"), "w") as f:
    #json.dump(train_report, f)

    # saving ner
    torch.save(ner, os.path.join(model_folder, "model.pickle"))




if __name__ == "__main__":
    import multiprocess.context as ctx
    from multiprocessing import set_start_method
    ctx._force_start_method('spawn')
    set_start_method("spawn")


    train_esn("word_gs_2L_bi_:spec_radius0.99-0.99", "raj's_library", None, True, "cpu")