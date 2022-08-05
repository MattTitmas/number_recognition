from typing import List, Callable
import numpy as np


def sigmoid(s: float | np.ndarray) -> float | np.ndarray:
    return 1 / (1 + np.exp(-s))


def sigmoid_prime(s: float | np.ndarray) -> float | np.ndarray:
    return s * (1 - s)


class NeuralNetwork(object):
    def __init__(self, input_nodes: int, output_nodes: int, hidden_layers: List[int],
                 learning_rate: float = 0.4):
        # 784,512,256,128,10
        self.__input_layer = input_nodes
        self.__hidden_layers = []
        self.__output_layer = output_nodes

        for i in hidden_layers:
            self.__hidden_layers.append(i)

        self.__learning_rate = learning_rate

        self.__weights = []
        self.__weights.append(np.random.randn(input_nodes, self.__hidden_layers[0]))
        for i in range(1, len(self.__hidden_layers)):
            self.__weights.append(np.random.randn(self.__hidden_layers[i - 1], self.__hidden_layers[i]))
        self.__weights.append(np.random.randn(self.__hidden_layers[-1], self.__output_layer))

        self.__activated = []
        for i in range(len(self.__hidden_layers)):
            self.__activated.append(np.array([]))

        self.__bias = []
        for i in range(len(self.__hidden_layers)):
            self.__bias.append(np.random.rand(1, self.__hidden_layers[i]))

        self.output_bias = np.random.rand(1, self.__output_layer)

    def forward_propagation(self, inp: np.ndarray) -> np.ndarray:
        prev = inp
        for i in range(len(self.__weights) - 1):
            dot_product = np.dot(prev, self.__weights[i])
            dot_product += self.__bias[i]
            self.__activated[i] = sigmoid(dot_product)
            prev = self.__activated[i]

        dot_output = np.dot(prev, self.__weights[-1])
        dot_output += self.output_bias
        prediction = sigmoid(dot_output)
        return prediction  # Output Layer

    def backward_propagation(self, x: np.ndarray, y: np.ndarray,
                             prediction: np.ndarray,
                             error_fun: Callable[[np.ndarray, np.ndarray], np.ndarray] = (lambda x, y: x - y)) -> None:
        errors = []
        deltas = []
        errors.append(error_fun(y, prediction))
        deltas.append(errors[0] * sigmoid_prime(prediction))

        for i in range(len(self.__weights) - 1, 0, -1):
            counting_forward = (len(self.__weights) - 1) - i
            error = deltas[-1].dot(self.__weights[i].T)
            delta = error * sigmoid_prime(self.__activated[-(counting_forward + 1)])
            errors.append(error)
            deltas.append(delta)

        self.__weights[0] += self.__learning_rate * (x.T.dot(deltas[-1]))
        for i in range(1, len(self.__weights) - 1):
            self.__weights[i] += self.__learning_rate * (self.__activated[i - 1].T.dot(deltas[-(i + 1)]))
        self.__weights[-1] += self.__learning_rate * (self.__activated[-1].T.dot(deltas[0]))

        self.output_bias += deltas[0]
        for i in range(len(self.__bias)):
            self.__bias[i] += deltas[-(i + 1)]

    def train(self, data: np.ndarray, expected: np.ndarray) -> None:
        prediction = self.forward_propagation(data)
        self.backward_propagation(data, expected, prediction)

    @property
    def get_weights(self):
        return self.__weights

    @property
    def get_bias(self):
        return self.__bias

    @property
    def get_learning_rate(self):
        return self.__learning_rate

    @property
    def get_input_layer(self):
        return self.__input_layer

    @property
    def get_hidden_layers(self):
        return self.__hidden_layers

    @property
    def get_output_layer(self):
        return self.__output_layer
