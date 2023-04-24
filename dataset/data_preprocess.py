import json
import os

from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

from spacy.lang.en import English

preprocessor = English()


# preprocess using spacy
def preprocess(sentence):
    # tokenize the sentence
    sentence = " ".join([str(token) for token in preprocessor(sentence.lower())])
    # remove punctuation and numbers
    sentence = " ".join(
        [word for word in sentence.split() if word.isalpha() and not word.isdigit()]
    )
    return sentence


# define and parse arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--file_path", type=str, required=True, help="The path of the file to be split."
    )
    parser.add_argument(
        "--train_size", type=float, default=0.8, help="The size of the train set."
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="The output directory."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # load the json file
    with open(args.file_path, "r") as f:
        data = json.load(f)

    # track max length
    max_len = 0
    max_len_sentence = ""

    # create a list of dictionaries
    dataset = []
    for value in data.values():
        sentence, label = value["joke"], value["NSFW"]
        sentence = preprocess(sentence)

        # skip empty sentences
        if not sentence:
            continue

        # update max length
        max_len = max(max_len, len(sentence.split()))
        if max_len == len(sentence.split()):
            max_len_sentence = sentence

        label = "NSFW" if label else "SFW"
        value = {"joke": sentence, "label": label}
        dataset.append(value)

    # split the data
    train, test = train_test_split(dataset, train_size=args.train_size, random_state=42)

    # write the data
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "train.json"), "w") as f:
        json.dump(train, f)
    with open(os.path.join(args.output_dir, "test.json"), "w") as f:
        json.dump(test, f)

    # print max length
    print(f"Max length: {max_len}")
    print(f"Max length sentence: {max_len_sentence}")
