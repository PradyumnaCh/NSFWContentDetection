import torch


class Config(object):
    def __init__(self):
        self.saved_dir = "./output"
        self.train_path = "./dataset/query_wellformedness/train.txt"
        self.test_path = "./dataset/query_wellformedness/test.txt"
        self.data_format = ['text', 'label']
        self.delimiter = '\t'
        self.pretrained_model_dir = None

        self.max_len = 256
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.pad_idx = 0
        self.embed_dim = 300
        self.hidden_dim = 10
        self.optim = 'sgd'

        self.num_epoch = 100
        self.lr = 0.001
        self.momentum = 0.9
        self.batch_size = 128
        self.valid_interval = 1
        self.random_seed = 64

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def add_attribute(self, attr_dict):
        for key, value in attr_dict.items():
            if key in self.__dict__:
                self.__dict__[key] = value


class FastTextConfig(Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'fasttext'


class TextRNNConfig(Config):
    def __init__(self):
        super().__init__()
        self.num_rnn_layers = 2
        self.dropout_prob = 0.8
        self.bidirectional = True
        self.model_type = 'textrnn'


CONFIG_MAP = {
    'fasttext': FastTextConfig,
    'textrnn': TextRNNConfig
}

if __name__ == '__main__':
    config = FastTextConfig()
    print()