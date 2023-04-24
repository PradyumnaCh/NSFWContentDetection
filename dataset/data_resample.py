# Script to resample (oversample or undersample) the given dataset
# to balance the number of samples in each class.

import json
import os

from argparse import ArgumentParser
from typing import List, Tuple

from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


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


def resample(data: List[List], over_sample=False, under_sample=False):
    # get the labels
    labels = [row[1] for row in data]

    # get the class distribution
    distribution = Counter(labels)
    print(f"Class distribution: {distribution}")

    # resample the dataset
    if over_sample:
        print("Oversampling the dataset...")
        resampler = RandomOverSampler(random_state=42)
        resamplers = resampler.fit_resample(data, labels)
        indices = resampler.sample_indices_
        print(f"Number of samples: {len(indices)}")
        data = [data[i] for i in indices]
        labels = [labels[i] for i in indices]
    elif under_sample:
        print("Undersampling the dataset...")
        resampler = RandomUnderSampler(random_state=42)
        data, labels = resampler.fit_resample(data, labels)

    # get the new class distribution
    distribution = Counter(labels)
    print(f"New class distribution: {distribution}")

    return data


def list_to_json(
    data: List[List], sentence_tag="joke", label_tag="label"
) -> List[dict]:
    json_data = []
    for row in data:
        json_data.append({sentence_tag: row[0], label_tag: row[1]})
    return json_data


# define and parse arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--file_path", type=str, required=True, help="The path of the unbalanced file."
    )
    sampler = parser.add_mutually_exclusive_group(required=True)
    sampler.add_argument(
        "--over_sample", action="store_true", help="Whether to oversample the dataset."
    )
    sampler.add_argument(
        "--under_sample",
        action="store_true",
        help="Whether to undersample the dataset.",
    )

    parser.add_argument(
        "--output_dir", type=str, default=".", help="The output directory."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # load the json file
    data = read_json_data(args.file_path)

    # resample the dataset
    data = resample(data, args.over_sample, args.under_sample)

    # convert the list to json
    data = list_to_json(data)

    # save the resampled dataset
    file_name = os.path.basename(args.file_path)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, file_name)
    with open(output_path, "w") as f:
        json.dump(data, f)
