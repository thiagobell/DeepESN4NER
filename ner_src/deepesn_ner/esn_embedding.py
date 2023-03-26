from typing import Union, List, Dict, Optional, Sequence
import os
import time
import tqdm
from flair.data import Dictionary, Sentence, Label, Token
import torch
import sys
import torch
import multiprocess.context as ctx
from multiprocessing import set_start_method, Process, Manager, Pool, get_start_method

from p_tqdm import p_map
import json
import logging
from esn_toolkit.reservoir import BaseReservoir, init_reservoir_from_param, Reservoir, DeepReservoir, Bi, Identity

from deepesn_ner.dataset import EmbeddedNERDataset, EfficientEmbeddedNERDataset
from deepesn_ner.config import esn_embeddings_folder

from functools import partial
def embed_dataset_with_reservoir(reservoir: BaseReservoir, dataset: EmbeddedNERDataset, split_name, transient: Union[float, int],
                                 device=None, num_cpu=1) -> EfficientEmbeddedNERDataset:
    """
    Embeds a dataset with a reservoir

    Takes a dataset embedded with flair or another language embedding and creates a dataset with ESN embeddings
    Args:
        reservoir: reservoir to use for embedding
        dataset: a dataset of NEREmbbeddings
        split_name: the name of the split
        transient: the size of the transient as an absolute length if int or ratio if float.
        device: if not none, embed using given device

    Returns: An Embedded repository in CPU

    Raises:
         InsufficientLengthError: if a sequence has length equal or shorter to/than the transient

    """
    if type(transient) == float and (transient < 0 or transient > 1):
        raise ValueError(f"invalid ratio provided as transient {transient}")
    print(f"device is: {device}")
    dataset = dataset.to("cpu")
    if device is not None:
        reservoir = reservoir.to(device)

    def embed_sentence_chunk_pool(sequence_chunk):
        return [reservoir.embed_sequence(sequence, transient, True).cpu() for sequence in sequence_chunk]

    target_sequences: List[torch.Tensor] = []  # contains the targets of each sequence
    sentences: List[Sentence] = []
    feature_sequence_chunks = [[]]
    max_chunk_size = 100

    for sentence, sequence, targets in dataset:
        # a (n x D) matrix where n is the length of the sequence and D is the size of the ESN embedding
        sequence = sequence.to(device)
        if len(feature_sequence_chunks[-1]) < max_chunk_size:
            feature_sequence_chunks[-1].append(sequence)
        else:
            # create a new chunk
            feature_sequence_chunks.append([sequence])
        target_sequences.append(targets)
        sentences.append(sentence)

    embedded_sequence_chunks = p_map(embed_sentence_chunk_pool, feature_sequence_chunks, num_cpus=num_cpu,
                                     desc=f"embedding chunks of split {split_name}")
    embedded_sequences = [sentence_embd for chunk in embedded_sequence_chunks for sentence_embd in chunk]

    if type(reservoir) == Bi:
        bidirectional = True
        print("embedded reservoir is bidirectional")
    else:
        bidirectional = False

    print(f"done embedding {split_name}")
    ds = EmbeddedNERDataset(sentences, embedded_sequences, target_sequences, dataset.tag_mapping, split_name,
                            bidirectional)
    return EfficientEmbeddedNERDataset.from_embedded_dataset(ds)


def embed_and_save_to_disk(res_path, split_name, dataset_path, transient, device_, save_folder, num_cpu,
                           trial_number: Optional[int]=None):
    print(f"calling embedding function  {device_}")
    sys.stdout.flush()
    print(f"loading dataset from {dataset_path}")
    dataset: EmbeddedNERDataset = EmbeddedNERDataset.from_pickle(dataset_path)
    dataset.split_name = split_name
    res = torch.load(res_path, map_location="cpu")


    # get_start_method
    print("START METHOD:")
    print(get_start_method(False))
    res = res.to(device_)

    #

    if trial_number is None:
        file_name = split_name + ".pickle"
    else:
        trial_number = int(trial_number)
        if trial_number < 0:
            raise ValueError("trial number must be positive")
        file_name = split_name + f".{trial_number}.pickle"
    path = os.path.join(save_folder, file_name)
    embedded_ds = embed_dataset_with_reservoir(res, dataset, dataset.split_name, transient, device_, num_cpu)
    torch.save(embedded_ds, path)
    print(f"{embedded_ds.split_name} is done")
    sys.stdout.flush()
    # releasing cuda memory
    if "cuda" in device_ or "gpu" in device_:
        torch.cuda.empty_cache()


def generate_dataset(name, reservoir_parameters: Dict, splits: Dict[str, str], transient: Union[float, int], device,
                     num_cpu, trial_number: Optional[int] = None):

    try:
        #ctx._force_start_method('spawn')

        set_start_method("spawn", force=True)
    except Exception as e:
        print("error setting spawn: " + str(e) + " ---> current context:" + get_start_method())


    embd_folder = os.path.join(esn_embeddings_folder, name)
    if not os.path.exists(embd_folder):
        os.makedirs(embd_folder)

    # generates reservoir
    res: BaseReservoir = init_reservoir_from_param(reservoir_parameters)

    # making sure the reservoir is in the cpu
    res = res.to("cpu")

    # saving reservoir to disk
    res_path = os.path.join(embd_folder, "res.pickle")
    # and its parameters
    with open(os.path.join(embd_folder, "res.param"), "w") as f:
        json.dump(reservoir_parameters, f)

    torch.save(res, res_path)
    del res
    for inx, (split_name, dataset_path) in enumerate(splits.items()):
        embed_and_save_to_disk(res_path, split_name, dataset_path, transient, device, embd_folder, num_cpu,
                               trial_number=trial_number)


def generate_dataset_parallel(name, reservoir_parameters: Dict, splits: Dict[str, str], transient: Union[float, int],
                              num_trials: int=1,
                              device_allocation = None, device_allocation_lock=None, devices=None, num_cpu=1):
    """
    Given a reservoir configuration, samples reservoir and
    Args:
        name: name to give to these embeddings
        reservoir_parameters:
        splits: a dictionary with split names as keys and path to pickled datasets as values
        transient: the size of the transient as an absolute length if int or ratio if float.
        num_trials: number of different reservoirs to generate with the given parameters and embed the dataset with
        device_allocation: a shared list with counters of free processing slots available in each device
        device_allocation_lock: lock that controls access to the list
        devices: a list of device names
        num_cpu: number of parallel processes this function can start
    Returns:

    """
    try:
        ctx._force_start_method('spawn')

        set_start_method("spawn", force=True)
    except Exception as e:
        print("error setting spawn: " + str(e) + " ---> current context:" + get_start_method())

    if type(num_trials) != int:
        raise TypeError("num trials must be int")
    if num_trials < 1:
        raise ValueError("num trials must be >= 1")
    device_inx = None
    try:
        # if None, use cpu
        if device_allocation is None:
            device = "cpu"
        else:
            # gets a device from list
            while device_inx is None:
                try:
                    print("getting lock")
                    with device_allocation_lock:
                        # get the gpu with the maximum number of free slots
                        max_avail_slots = max(device_allocation)
                        if max_avail_slots == 0:
                            # no free slots
                            raise ValueError
                        device_inx = device_allocation.index(max_avail_slots)

                        device_allocation[device_inx] -= 1
                        print(f"acquired lock on device inx: {device_inx}")
                    print(device_allocation)
                except ValueError:
                    # no device is free
                    print("could not find any free device. waiting 60seconds")
                    time.sleep(120)

            device = devices[device_inx]
            print(f"acquired device: {device}")
        sys.stdout.flush()
        for trial in range(num_trials):
            generate_dataset(name, reservoir_parameters, splits, transient, device, num_cpu, trial_number=trial)
    except Exception as e:
        logging.critical(f"{name}: Exception raised: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # free device
        with device_allocation_lock:
            device_allocation[device_inx] += 1
        print(f"releasing lock on device {device}")
        print(device_allocation)
        torch.cuda.empty_cache()


def identity(input_dim):
    rs = Identity(input_dim)
    return rs


def vanilla_res(input_dim, reservoir_size, bidirectional=False):
    rs = Reservoir(input_dim, True, reservoir_size=reservoir_size, input_weight_scale=0.25, input_connectivity=0.5,
                   spectral_radius=0.7,
                   reservoir_connectivity=0.5, reservoir_weight_scale=0.5,
                   leaking_rate=0.75, available_activation_functions=["tanh"], dry_run=True)
    if bidirectional:
        rs = Bi(rs)
    return rs


def genESN(input_dim, reservoir_sizes: List, bidirectional=False, hyper_param: Dict[str, Union[float, List[float]]]=None):
    """

    Args:
        input_dim:
        reservoir_sizes:
        bidirectional:
        leaking_rate:
        reservoir_connectivity:

    Returns:


    default values for hyperparam



    """

    res = []

    default_values_hyper_param = {
        "leaking_rate": 0.75,
        "reservoir_connectivity": 0.5,
        "spectral_radius": 0.7,
        "input_connectivity": 0.5,
        "reservoir_topology": "sparse"
    }

    default_values_hyper_param.update(hyper_param)

    for param in default_values_hyper_param.keys():
        if type(default_values_hyper_param[param]) == str or not isinstance(default_values_hyper_param[param], Sequence):
            # if its a string or not a sequence, repeat values for all reservoir layers
            default_values_hyper_param[param] = [default_values_hyper_param[param]]*len(reservoir_sizes)
        else:
            if len(default_values_hyper_param[param]) != len(reservoir_sizes):
                raise ValueError(f"number of values for {param} does not match number of layers")
            else:
                default_values_hyper_param[param] = default_values_hyper_param[param]
    print(default_values_hyper_param)
    for inx, res_size in enumerate(reservoir_sizes):
        input_size = input_dim if inx == 0 else reservoir_sizes[inx-1]
        r = Reservoir(input_size, True, reservoir_size=res_size, input_weight_scale=0.25,
                      input_connectivity=default_values_hyper_param["input_connectivity"][inx],
                      spectral_radius=default_values_hyper_param["spectral_radius"][inx],
                      reservoir_connectivity=default_values_hyper_param["reservoir_connectivity"][inx], reservoir_weight_scale=0.5,
                      leaking_rate=default_values_hyper_param["leaking_rate"][inx], available_activation_functions=["tanh"],
                      dry_run=True, reservoir_topology=default_values_hyper_param["reservoir_topology"][inx])
        res.append(r)

    if len(reservoir_sizes) > 1:
        # deep
        res = DeepReservoir(res, "cpu")
    else:
        res = res[0]
    
    if bidirectional:
        res = Bi(res)

    return res


def lr_gs():
    """ calc parameters for a learning rate gridsearch"""

    layers = [1024, 1024]
    leaking_rate_values = [0.55, 0.75, 0.95]
    base_name = "word_gs_2L"
    params = {}

    for lk1 in leaking_rate_values:
        for lk2 in leaking_rate_values:

            hyper_param = {
                "leaking_rate": [lk1, lk2]
            }
            params[base_name+"_bi_"+f":{lk1}-{lk2}"] = genESN(300, layers, bidirectional=True,
                                                              hyper_param=hyper_param).param_dict()

    return params


def reservoir_connectivity_grid_search():
    layers = [1024, 1024]
    connectivity_values = [0.1, 0.5, 0.9]
    params = {}
    base_name = "word_gs_2L"
    for cv1 in connectivity_values:
        for cv2 in connectivity_values:
            hyper_param = {
                "reservoir_connectivity": [cv1, cv2],
                "leaking_rate": [0.95, 0.55]
            }
            params[base_name + "_bi_" + f":res_conn{cv1}-{cv2}"] = genESN(300, layers, bidirectional=True,
                                                                          hyper_param=hyper_param).param_dict()
    return params


def random_search(num_experiments, reservoir_connectivity_values, leaking_rates, spectral_radii, input_connectivity_values):
    import random
    params = {}
    for i in range(num_experiments):
        lk = random.choice(leaking_rates)
        sr = random.choice(spectral_radii)
        ic = random.choice(input_connectivity_values)
        rc = random.choice(reservoir_connectivity_values)

        param = {
            "leaking_rate": lk,
            "reservoir_connectivity": rc,
            "spectral_radius": sr,
            "input_connectivity": ic
        }
        params[f"random_search_1024-1024_ex{i}"] = genESN(300, [1024, 1024], bidirectional=True,
                                                          hyper_param=param).param_dict()
    return params


def grid_search(base_name, reservoir_connectivity_values, leaking_rates, spectral_radii, input_connectivity_values):
    params = {}
    layers = [1024, 512, 512]
    for rc in reservoir_connectivity_values:
        for lk in leaking_rates:
            for sr in spectral_radii:
                for ic in input_connectivity_values:
                    param = {
                        "leaking_rate": lk,
                        "reservoir_connectivity": rc,
                        "spectral_radius": sr,
                        "input_connectivity": ic,
                        "reservoir_topology": "sparse"
                    }
                    params[f"{base_name}_GS_{'-'.join([str(l) for l in layers])}_rc{rc}_lk{lk}_sr{sr}_ic{ic}"] = genESN(
                        300, layers, bidirectional=True, hyper_param=param).param_dict()
    return params


def grid_search_permutated(base_name, architectures: List[List[int]],
                           leaking_rates, spectral_radii, input_connectivity_values):
    params = {}

    for layers in architectures:
        for lk in leaking_rates:
            for sr in spectral_radii:
                for ic in input_connectivity_values:
                    param = {
                        "leaking_rate": lk,
                        "spectral_radius": sr,
                        "input_connectivity": ic,
                        "reservoir_topology": "permutated"
                    }
                    params[f"{base_name}_GS_{'-'.join([str(l) for l in layers])}_permutated_lk{lk}_sr{sr}_ic{ic}"] = genESN(
                        300, layers, bidirectional=True, hyper_param=param).param_dict()
    return params



def reservoir_spectral_radius_grid_search():
    layers = [1024, 1024]
    spectral_radius = [0.5, 0.7, 0.99]
    base_name = "word_gs_2L"
    params = {}
    for sr1 in spectral_radius:
        for sr2 in spectral_radius:
            hyper_param = {
                "reservoir_connectivity": [0.1, 0.1],
                "leaking_rate": [0.95, 0.55],
                "spectral_radius": [sr1, sr2]
            }
            params[base_name + "_bi_" + f":spec_radius{sr1}-{sr2}"] = genESN(300, layers, bidirectional=True,
                                                                             hyper_param=hyper_param).param_dict()
    return params


if __name__ == "__main__":
    """
    try:
        torch.multiprocessing.set_start_method('spawn')
        set_start_method("spawn")
        ctx._force_start_method('spawn')
    except:
        pass
    """

    try:
        #ctx._force_start_method('spawn')

        set_start_method("spawn", force=True)
    except Exception as e:
        print("error setting spawn: " + str(e) + " ---> current context:" + get_start_method())


    """
    train_dataset_lang_model: EmbeddedNERDataset = torch.load("../../embeddings/germeval_word_embd_train.pickle",
                                                              map_location="cpu")
    dev_dataset_lang_model: EmbeddedNERDataset = torch.load("../../embeddings/germeval_word_embd_dev.pickle",
                                                            map_location="cpu")
    test_dataset_lang_model: EmbeddedNERDataset = torch.load("../../embeddings/germeval_word_embd_test.pickle",
                                                             map_location="cpu")
    """


    def dataset_splits(name_dataset):
        return {"train": f"../../embeddings/{name_dataset}_train.pickle",
               "test": f"../../embeddings/{name_dataset}_test.pickle",
               "dev": f"../../embeddings/{name_dataset}_dev.pickle"}


    #dataset_name = "germeval_word_embd"
    dataset_name = "conll2003_word"

    slot_per_gpu = 1
    gpus = [0, 2, 3, 4, 5]
    num_gpu = len(gpus)
    devices = [f"cuda:{inx}" for inx in gpus]
    mg = Manager()
    device_access = mg.list([slot_per_gpu] * len(devices))
    device_access_lock = mg.Lock()

    #param_configs = reservoir_spectral_radius_grid_search()
    """
    param_configs = grid_search(dataset_name, reservoir_connectivity_values=[0.1],
                                leaking_rates=[0.7],
                                spectral_radii=[0.7],
                                input_connectivity_values=[0.5])
    
    """
    param_configs = grid_search_permutated(dataset_name, architectures=[[1024, 512, 512], [1024, 1024], [2048]],
                                leaking_rates=[0.7],
                                spectral_radii=[0.7, 0.9],
                                input_connectivity_values=[0.5])

    print(f"{len(param_configs)} configurations to run")
    print(param_configs.keys())
    print("starting subprocesses")

    processes = []
    process_names = []

    SERIAL_DEBUG = False
    num_trials = 1

    for name, config in param_configs.items():
        if SERIAL_DEBUG:
            generate_dataset_parallel(name, config, dataset_splits(dataset_name), 0, num_trials, device_access,
                                      device_access_lock, devices, 1)
        else:
            # we use processes here so we can create them as non daemonic and can have pools inside them :)
            p = Process(target=generate_dataset_parallel,
                        args=(name, config, dataset_splits(dataset_name), 0, num_trials, device_access,
                              device_access_lock, devices, 5),
                        daemon=True)
            p.start()
            process_names.append(name)
            processes.append(p)

    if not SERIAL_DEBUG:
        for inx, p in enumerate(processes):
            p.join()

            if p.exitcode >= 0:
                logging.info(f"{p}:{process_names[inx]} finished successfully")
            else:
                logging.critical(f"{p}:{process_names[inx]} finished with exit code {p.exitcode}")

    print("\n".join(param_configs.keys()))
