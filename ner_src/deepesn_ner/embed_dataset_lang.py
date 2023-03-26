"""
This module reads the dataset and generete series of labelled token embeddings organized by sentence

input: a tsv dataset
output: a Dataset object where each sample corresponds to a sentence in the original dataset with token embeddings. The
features are described by a (ts x e) matrix where ts is the number of tokens in the sentence and e is the embedding
dimension
"""
from multiprocessing import set_start_method
import multiprocess.context as ctx

import datasets

from multiprocessing import Pool
import numpy

from typing import List, Callable
from functools import partial

from flair.data import Corpus, Sentence, Dictionary, FlairDataset, Token
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
import flair

from torch.utils.data import Dataset

#flair.device = "cuda"
print("loading module")
from p_tqdm import p_imap
import torch

import tqdm

from deepesn_ner.dataset import EmbeddedNERDataset


def embed_ner_dataset(sentences: FlairDataset, embedder: Callable[[str], StackedEmbeddings], tag_dictionary: Dictionary, split_name, devices=["cpu"]) -> \
        EmbeddedNERDataset:
    """
    Embeds one sentence
    Args:
        sentences:
        embedder:
        tag_dictionary: a dictionary mapping NER tags to an index or code

    Returns:

    """
    features = []
    targets = []

    embed_fun = partial(embed_chunk_sentence, embedder=embedder, tag_dictionary=tag_dictionary)
    sentence_list: List[Sentence] = []
    #p = Pool(4)
    #print("starting map")
    #embeddings = p.map(embed_fun, sentences)

    #p.close()

    #p.join()
    #print("finishing map")
    ctx._force_start_method('spawn')
    chunks = numpy.array_split(sentences, len(devices))
    chunk_device = []
    for inx in range(len(chunks)):
        device_inx = inx % len(devices)
        chunk_device.append(devices[device_inx])

    for embd_chunk in p_imap(embed_fun, chunks, chunk_device, num_cpus=len(devices)):
        #for embd_sen in embeddings:
        for embd_sen in embd_chunk:
            sentence, s_features, s_targets = embd_sen
            features.append(s_features.to(s_targets.device))
            targets.append(s_targets)
            sentence_list.append(sentence)

    return EmbeddedNERDataset(sentence_list, features, targets, tag_dictionary, split_name, False)


def embed_chunk_sentence(sentences, device, embedder, tag_dictionary: Dictionary):
    results = []
    embedder = embedder(device)
    print(f"done loading embedder on device {device}")
    for sentence in sentences:
        results.append(embed_sentence(sentence, embedder, tag_dictionary))
    return results


def embed_sentence(sentence: Sentence, embedder, tag_dictionary: Dictionary):
    embedder.embed(sentence)
    token_embeddings = torch.cat([token.get_embedding().unsqueeze(0) for token in sentence], dim=0).cpu()
    for token in sentence:
        token.clear_embeddings()
    token_tags = list(map(lambda token: token.get_tag("ner").value, sentence))
    token_targets: torch.Tensor = torch.tensor(tag_dictionary.get_idx_for_items(token_tags)).cpu()

    return sentence, token_embeddings, token_targets



def embed_corpus(corpus: Corpus,  language="en", embedding_type="word"):
    """
    Embeds a dataset from one of flair's NLPTasks.
    Args:
        corpus: a Corpus object with NER tags "ner"
        language: language to use en/de
        embedding_type: word or word+flair

    Returns:

    """

    tag_type = 'ner'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    def get_flairword_embedder(device):
        flair.device = device
        embedder = StackedEmbeddings(
            embeddings=[WordEmbeddings(language), FlairEmbeddings(language+"-forward"), FlairEmbeddings(language+"-backward")])
        return embedder

    def get_word_embedder(device):
        flair.device = device
        embedder = StackedEmbeddings(
            embeddings=[WordEmbeddings(language)])
        return embedder

    if embedding_type == "word+flair":
        embedder = get_flairword_embedder

    if embedding_type == "word":
        embedder = get_word_embedder

    num_gpu = 3

    train_dataset = embed_ner_dataset(corpus.train, get_word_embedder, tag_dictionary, "train",
                                      devices=[f"cuda:{inx}" for inx in range(num_gpu)])
    dev_dataset = embed_ner_dataset(corpus.dev, get_word_embedder, tag_dictionary, "dev",
                                    devices=[f"cuda:{inx}" for inx in range(num_gpu)])
    test_dataset = embed_ner_dataset(corpus.test, get_word_embedder, tag_dictionary, "test",
                                     devices=[f"cuda:{inx}" for inx in range(num_gpu)])

    dataset_name = corpus.name+"_"+embedding_type
    torch.save(train_dataset, f"../../embeddings/{dataset_name}_train.pickle")
    torch.save(dev_dataset, f"../../embeddings/{dataset_name}_dev.pickle")
    torch.save(test_dataset, f"../../embeddings/{dataset_name}_test.pickle")


def embed_germeval():
    corpus: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.GERMEVAL)
    embed_corpus(corpus, language="de", embedding_type="word")


def embed_from_datasets_library(dataset_name, language):
    """Takes a dataset from the datasets package and embeds it"""
    corpus = generate_corpus_from_datasets(dataset_name)
    embed_corpus(corpus, language)


def generate_corpus_from_datasets(dataset_name="conll2003") -> Corpus:
    ds = datasets.load_dataset(dataset_name)

    def create_flair_ds(split_data) -> List[Sentence]:
        sentences = []
        for sentence_data in split_data:
            ner_tags = sentence_data["ner"]
            words = sentence_data["words"]
            st = Sentence()

            for word, ner_tag in zip(words, ner_tags):
                tk = Token(word)
                tk.add_tag("ner", ner_tag)
                st.add_token(tk)
            sentences.append(st)
        return sentences

    train = create_flair_ds(ds["train"])
    dev = create_flair_ds(ds["validation"])
    test = create_flair_ds(ds["test"])
    return Corpus(train, dev, test, dataset_name)


if __name__ == "__main__":
    embed_from_datasets_library("conll2003", "en")