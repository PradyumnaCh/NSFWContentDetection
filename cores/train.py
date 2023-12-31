import os
import argparse
import yaml
import time

from torch.utils.data import DataLoader
from cores.config import *
from cores.datahelper import TextDataset, load_word2vec, save_vocab
from cores.models import MODEL_MAP, save_checkpoint
from cores.config import CONFIG_MAP
from sklearn import metrics
from tqdm import tqdm
from cores.logger import init_logger, logger, seed_everything


def eval(test_iter, model, device):
    avg_loss = 0
    predicts = []
    actuals = []
    model.eval()

    tqdm_bar = tqdm(
        enumerate(test_iter), total=len(test_iter), desc="Eval", leave=False, position=0
    )
    for idx, batch in tqdm_bar:
        sent, sent_lens, labels = batch
        input_ids, seq_lens, label_ids = batch
        if device == "cuda":
            input_ids = input_ids.to(device)
            # seq_lens = seq_lens.to(device)
            label_ids = label_ids.to(device)
        loss, probs = model(input_ids, seq_lens.tolist(), label_ids)

        avg_loss += loss.item()
        predicts += [y.argmax().item() for y in probs]
        actuals += labels.tolist()

    metric = metrics.classification_report(
        actuals, predicts, target_names=test_iter.dataset.label_set
    )
    acc_score = metrics.accuracy_score(actuals, predicts)
    macro_f1_score = metrics.f1_score(actuals, predicts, average="macro")
    logger.info(
        f"VALID - AVG Loss: {avg_loss / len(test_iter):.6f}; Accurancy: {acc_score:.4f}; F1"
        f": {macro_f1_score:.4f}"
    )

    logger.info(metric)
    return metric, acc_score, macro_f1_score, f"{avg_loss / len(test_iter):.6f}"


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_total_time(start_time):
    end = time.time()
    hours, rem = divmod(end - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def train(config):
    # Read YAML file
    with open(config, "r") as stream:
        config_loaded = yaml.safe_load(stream)
        opts = CONFIG_MAP[config_loaded["model_type"]]()
        opts.add_attribute(config_loaded)

    if not os.path.exists(opts.saved_dir):
        os.makedirs(opts.saved_dir)
    init_logger(log_file=opts.saved_dir + "/{}.log".format(opts.model_type))
    seed_everything(opts.random_seed)

    if not opts.pretrained_embedding is None:
        logger.info("Loading pretrained word2vec ...")
        vocab, vectors = load_word2vec(opts)
    else:
        vocab, vectors = None, None

    logger.info("Loading TRAIN dataset ...")
    train_dataset = TextDataset(
        opts.train_path,
        data_format=opts.data_format,
        delimiter=opts.delimiter,
        vocab=vocab,
        label_set=None,
        max_len=opts.max_len,
        model_type=opts.model_type,
        pad_token=opts.pad_token,
        unk_token=opts.unk_token,
    )

    opts.vocab_size = len(train_dataset.vocab)
    opts.num_labels = len(train_dataset.label_set)
    opts.pad_idx = train_dataset.vocab[opts.pad_token]
    logger.info("Loading TEST dataset ...")
    test_dataset = TextDataset(
        opts.test_path,
        data_format=opts.data_format,
        model_type=opts.model_type,
        delimiter=opts.delimiter,
        vocab=train_dataset.vocab,
        label_set=train_dataset.label_set,
        max_len=opts.max_len,
        pad_token=opts.pad_token,
        unk_token=opts.unk_token,
    )

    logger.info("Saving vocab into json file %s", str(opts.saved_dir + "/vocab.json"))
    save_vocab(
        opts.saved_dir + "/vocab.json",
        vocab=train_dataset.vocab,
        label_set=train_dataset.label_set,
        pad_token=opts.pad_token,
        unk_token=opts.unk_token,
    )

    model = MODEL_MAP[opts.model_type](opts, vectors)

    if opts.pretrained_model_dir is not None:
        model_checkpoint = torch.load(opts.pretrained_model_dir + "/model.model")
        model.load_state_dict(model_checkpoint)

    logger.info("=" * 30 + "MODEL SUMMARY" + "=" * 30)
    logger.info(model)

    if opts.device == "cuda":
        model.cuda()

    if opts.optim == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=opts.lr
        )
    elif opts.optim == "sgd":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opts.lr,
            momentum=opts.momentum,
        )
    else:
        raise Exception(f"{opts.optim} is not Found !!")
    if opts.decay_steps == -1:
        opts.decay_steps = int(len(train_dataset) / opts.batch_size)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=opts.decay_steps, gamma=opts.decay_rate
    )

    best_score = float("-inf")
    best_epoch = float("-inf")
    train_iter = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    valid_iter = DataLoader(
        test_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    train_time = time.time()
    for epoch in range(opts.num_epoch):
        logger.info(f"{'='*30}Epoch: {epoch + 1}/{opts.num_epoch}{'='*30}")
        start_time = time.time()
        train_avg_loss = 0
        predicts = []
        actuals = []
        model.train()
        model.to(opts.device)
        tqdm_bar = tqdm(
            enumerate(train_iter),
            total=len(train_iter),
            desc="Train",
            leave=False,
            position=0,
        )
        for idx, batch in tqdm_bar:
            optimizer.zero_grad()
            input_ids, seq_lens, label_ids = batch
            if opts.device == "cuda":
                input_ids = input_ids.to(opts.device)
                label_ids = label_ids.to(opts.device)
            loss, probs = model(input_ids, seq_lens.tolist(), label_ids)
            loss.backward()
            clip_gradient(model, 1e-1)
            optimizer.step()
            scheduler.step()
            train_avg_loss += loss.item()
            predicts += [y.argmax().item() for y in probs]
            actuals += label_ids.tolist()
        train_acc_score = metrics.accuracy_score(actuals, predicts)
        train_f1_score = metrics.f1_score(actuals, predicts, average="macro")
        logger.info(
            f"INFO  - Lr: {get_lr(optimizer)}; Time: {get_total_time(start_time)}"
        )
        logger.info(
            f"TRAIN - AVG Loss: {train_avg_loss / len(train_iter):.6f}; Accurancy: {train_acc_score:.4f}; F1"
            f": {train_f1_score:.4f}"
        )
        if epoch % opts.valid_interval == 0:
            metric, eval_acc_score, eval_f1_score, eval_avg_loss = eval(
                valid_iter, model, opts.device
            )
            if eval_f1_score > best_score:
                save_checkpoint(
                    opts.saved_dir,
                    model,
                    epoch,
                    f"{train_avg_loss / len(train_iter):.6f}",
                    train_acc_score,
                    train_f1_score,
                    eval_avg_loss,
                    eval_acc_score,
                    eval_f1_score,
                    metric,
                )
                best_score = eval_f1_score
                best_epoch = epoch + 1

    logger.info("\n" + "=" * 70)
    logger.info(
        f"RESULT  - Best Score: {best_score:.4f}; Best Epoch: {best_epoch}; Time: {get_total_time(train_time)}"
    )
    return best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=None, type=str, required=True, help="The config file."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise ValueError(
            "Config file ({}) not exists or is empty !!!".format(args.config)
        )

    # Read YAML file
    with open(args.config, "r") as stream:
        config_loaded = yaml.safe_load(stream)
        config = CONFIG_MAP[config_loaded["model_type"]]()
        config.add_attribute(config_loaded)
    train(config)
