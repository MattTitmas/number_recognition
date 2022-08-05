from neural_network import NeuralNetwork
import numpy as np
import csv
from tqdm import tqdm

if __name__ == '__main__':
    AI = NeuralNetwork(784, 10, [512, 256, 128])

    epochs = 3
    for i in tqdm(range(epochs)):
        with open('data/train.csv', 'r') as training_data:
            datareader = csv.reader(training_data)
            for c, row in enumerate(tqdm(datareader, total=60000, leave=False)):
                data = np.where(np.array(row[1:]).astype(int) < 127, 0, 1).reshape(1, 28*28)
                expected = np.zeros(10)
                expected[int(row[0])] = 1
                AI.train(data, expected)

    for c, i in enumerate(AI.get_weights):
        np.save('data/weights/weight'+str(c)+'.npy', i)


    for c, i in enumerate(AI.get_bias):
        np.save('data/weights/bias'+str(c)+'.npy', i)

