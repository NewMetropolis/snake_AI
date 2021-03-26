# import snake
import numpy as np


class HopfieldNetwork:

    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.nodes = np.zeros(self.n_nodes, dtype='int')
        self.weights = np.zeros([self.n_nodes, self.n_nodes])
        self.bias = np.zeros(n_nodes)

    def calculate_energy(self, state):

        energy = 0
        for i in range(self.n_nodes):
            energy -= self.bias[i] * state[i]
            for j in range(i + 1, self.n_nodes):
                energy -= self.weights[i, j] * state[i] * state[j]

        return energy

    def train(self, training_points, mode='all_edges', start=0, stop=0, skip=0, start_2=0, stop_2=0):

        n = training_points.shape[0]
        if mode == 'all_edges':
            for i in range(start, stop):
                for j in range(i + 1, stop):
                    weight = 0
                    for point in training_points:
                        weight += (point[i] - 0.5) * (point[j] - 0.5)
                    weight = 4 * weight / n
                    self.weights[i, j] = self.weights[j, i] = weight
                bias = 0
                for point in training_points:
                    bias += point[i] - 0.5
                bias = 2 * bias / n
                self.bias[i] = bias

        if mode == 'ith':
            for i in range(start, stop):
                j = i + skip
                bias = 0
                weight = 0
                for point in training_points:
                    weight += (point[i] - 0.5) * (point[j] - 0.5)
                    bias += point[i] - 0.5
                weight = 4 * weight / n
                self.weights[i, j] = self.weights[j, i] = weight
                bias = 2 * bias / n
                self.bias[i] = bias

        if mode == 'one_to_all':
            for i in range(start, stop):
                for j in range(start_2, stop_2):
                    weight = 0
                    for point in training_points:
                        weight += (point[i] - 0.5) * (point[j] - 0.5)
                    weight = 4 * weight / n
                    self.weights[i, j] = self.weights[j, i] = weight
                bias = 0
                for point in training_points:
                    bias += point[i] - 0.5
                bias = 2 * bias / n
                self.bias[i] = bias

    def flip_a_node(self, i):
        energy_gap = 0
        for j in range(self.n_nodes):
            if i != j:
                energy_gap += self.weights[i, j] * self.nodes[j]
        energy_gap += self.bias[i]

        if energy_gap >= 0:
            self.nodes[i] = 1
        else:
            self.nodes[i] = 0

        return

    def recall(self, _input, epochs=1, mode='total'):
        self.nodes = _input
        for epoch in range(epochs):
            if mode == 'total':
                for i in range(self.n_nodes):
                    self.flip_a_node(i)
            if mode == 'move':
                for i in range(4, 8):
                    self.flip_a_node(i)

        return self.nodes
