import csv
import argparse
from typing import List

import numpy as np
from tqdm import tqdm

from neural_network import NeuralNetwork


def main():
    parser = argparse.ArgumentParser(description='Train a number recognition model.')
    parser.add_argument('-t', '--train', type=str, required=True,
                        help='Location of training file.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory of output weights.')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=3,
                        help='Number of epochs to train with.')
    parser.add_argument('-hn', '--hidden_nodes', nargs='+', type=List[int], required=False, default=[512, 256, 128],
                        help='Number of hidden nodes per layer.')

    args = parser.parse_args()
    number_recognition_model = NeuralNetwork(784, 10, args.hidden_nodes)

    for _ in tqdm(range(args.epochs)):
        with open(args.train, 'r') as training_data:
            datareader = csv.reader(training_data)
            for c, row in enumerate(tqdm(datareader, total=60000, leave=False)):
                data = np.where(np.array(row[1:]).astype(int) < 127, 0, 1).reshape(1, 28 * 28)
                expected = np.zeros(10)
                expected[int(row[0])] = 1
                number_recognition_model.train(data, expected)

    for c, i in enumerate(number_recognition_model.get_weights):
        np.save(args.output + str(c) + '.npy', i)

    for c, i in enumerate(number_recognition_model.get_bias):
        np.save(args.output + str(c) + '.npy', i)


if __name__ == '__main__':
    main()
