import os
import torch
import json

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from cores.logger import logger


def save_vocab(file_path, vocab, label_set, pad_token, unk_token):
    data = {
        "vocab": vocab,
        "label_set": label_set,
        "pad_token": pad_token,
        "unk_token": unk_token,
    }
    json.dump(data, open(file_path, "w", encoding="utf-8"))


def read_delimited_data(file_path, delimiter="\t"):
    datasets = []
    if delimiter == "\\t":
        delimiter = "\t"
    with open(file_path, "r", encoding="utf-8") as f:
        rows = f.readlines()
        for row in rows:
            cols = row.split(delimiter)
            if len(cols) != 0:
                datasets.append(cols)
    return datasets


def read_json_data(file_path, sentence_tag="joke", label_tag="label"):
    datasets = []
    # open the json file
    with open(file_path) as f:
        data = json.load(f)

    # iterate over the objects in the array
    for obj in data:
        # extract the values in the sentence_tag and label_tag keys
        sentence = obj.get(sentence_tag)
        label = obj.get(label_tag)
        datasets.append([sentence, label])
    return datasets


def read_data(file_path, delimiter="\t"):
    if file_path.endswith("json"):
        return read_json_data(file_path)
    else:
        return read_delimited_data(file_path, delimiter)


def load_word2vec(opts):
    with open(opts.pretrained_embedding, "r", encoding="utf-8") as f:
        lines = f.readlines()
        vocab_size = len(lines)
        embed_dim = len(lines[0].split(" ")) - 1
        vocab = []
        vectors = []
        if opts.pad_token is not None:
            vocab.append(opts.pad_token)
            vocab_size += 1
            vectors.append(torch.zeros([embed_dim], dtype=torch.float))
        if opts.unk_token is not None:
            vocab.append(opts.unk_token)
            vocab_size += 1
            vectors.append(torch.rand([embed_dim], dtype=torch.float))
        for line in tqdm(lines, total=vocab_size, leave=False, position=0):
            line = line.split(" ")
            token_vector = list(
                map(
                    lambda t: float(t),
                    filter(lambda n: n and not n.isspace(), line[1:]),
                )
            )
            vocab.append(line[0])
            vectors.append(torch.tensor(token_vector, dtype=torch.float))
    return vocab, torch.stack(vectors)


@dataclass
class Example:
    input_ids: list
    label_id: int
    seq_len: int
    raw_text: str = None
    raw_label: str = None


class TextDataset(Dataset):
    def __init__(
        self,
        file_path,
        model_type: str,
        data_format: list = ["text", "label"],
        delimiter="\t",
        vocab=None,
        label_set=None,
        max_len=40,
        pad_token="<pad>",
        unk_token="<unk>",
    ):
        self.file_path = file_path
        self.model_type = model_type
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.max_len = max_len
        self.data_format = data_format

        if vocab is None:
            self.init_vocab = True
            self.vocab = {
                self.pad_token: 0,
                self.unk_token: 1,
            }
        else:
            self.init_vocab = False
            self.vocab = vocab

        self.init_label, self.label_set = (
            (True, []) if label_set is None else (False, label_set)
        )

        dataset_name = os.path.basename(self.file_path).split(".")[0].strip()
        cached_dir = os.path.dirname(self.file_path)
        self.cached_file = (
            cached_dir + f"/cached_file_{self.model_type}_{self.max_len}_{dataset_name}"
        )
        if os.path.exists(self.cached_file):
            self.examples = []
            self.load_cached_file()
        else:
            self.examples = self.create_examples(read_data(file_path, delimiter))
            self.cache_dataset()

    @property
    def pad_id(self):
        return self.vocab[self.pad_token]

    @property
    def unk_id(self):
        return self.vocab[self.unk_token]

    def convert_tokens_to_ids(self, text):
        tokens = text.split()
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                if self.init_vocab:
                    self.vocab[token] = len(self.vocab)
                    token_ids.append(len(self.vocab) - 1)
                else:
                    token_ids.append(self.vocab[self.unk_token])
        return token_ids

    def convert_label_to_id(self, label):
        if label in self.label_set:
            return self.label_set.index(label)
        else:
            if self.init_label:
                self.label_set.append(label)
                return len(self.label_set) - 1
            else:
                raise Exception(f"Label {label} is not found !!")

    def preprocess(self, text):
        return text

    def cache_dataset(self):
        data = {
            "vocab": self.vocab,
            "label_set": self.label_set,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "max_len": self.max_len,
            "data_format": self.data_format,
            "examples": self.examples,
        }
        logger.info("\tSaving Dataset into cached file %s", self.cached_file)
        torch.save(data, self.cached_file)

    def load_cached_file(self):
        logger.info("\tLoad Dataset from cached file %s", self.cached_file)
        data = torch.load(self.cached_file)
        for key, value in data.items():
            if key in self.__dict__:
                self.__dict__[key] = value

    def create_examples(self, dataset):
        examples = []
        pad_id = self.vocab[self.pad_token]
        text_id = self.data_format.index("text")
        label_id = self.data_format.index("label")

        for example in tqdm(dataset):
            examples.append(
                self.create_single_example(example, pad_id, text_id, label_id)
            )

        return examples

    def create_single_example(self, cols, pad_id, text_id, label_id):
        raw_text = cols[text_id].strip()
        raw_label = cols[label_id]

        ex_input_ids = self.convert_tokens_to_ids(raw_text)
        ex_label_id = self.convert_label_to_id(raw_label)

        ex_length = len(ex_input_ids)

        return Example(input_ids=ex_input_ids, label_id=ex_label_id, seq_len=ex_length)

    @staticmethod
    def collate_fn(batch):
        # use torch pad_sequence to create batches
        all_input_ids, all_lens, all_labels = zip(*batch)
        all_input_ids = torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True)
        all_lens = torch.stack(all_lens)
        all_labels = torch.stack(all_labels)
        return all_input_ids, all_lens, all_labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        ex_input_tensor = torch.tensor(example.input_ids, dtype=torch.long)
        ex_label_tensor = torch.tensor(example.label_id, dtype=torch.long)
        ex_seq_length_tensor = torch.tensor(example.seq_len, dtype=torch.long)
        return ex_input_tensor, ex_seq_length_tensor, ex_label_tensor


# test dataloader if run as main
if __name__ == "__main__":
    dataset = TextDataset("dataset/query_wellformedness/dev.txt")
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=dataset.get_collate_fn
    )
    for batch in dataloader:
        print(batch)
