import math
import numpy as np
from scipy.special import expit


class RBM:
    # Restricted Boltzman Machine (RBM) with d visible and m hidden units connected edges with weight matrix W.
    # Once trained, the machine can sample hidden unit states given visible units et vice-versa.
    # For learning weights we use a contrastive divergence algorithm.

    def __init__(self, d, m, alpha=0.1, classifier=False, k=0):
        # N visible units.
        self.d = d
        # N hidden units.
        self.m = m
        # Visible units.
        self.visible = np.empty(self.d)
        # Hidden units.
        self.hidden = np.empty(self.m)
        # Bias for visible units.
        self.bias_v = np.zeros(d)
        # Bias for hidden units.
        self.bias_h = np.zeros(m)
        # Weight matrix.
        self.weights = np.random.standard_normal([self.d, self.m]) / np.sqrt(self.d * self.m)
        # Conditional probability assigned to visible units.
        self.prob_v = np.empty(d)
        # Conditional probability assigned to hidden units.
        self.prob_h = np.empty(m)
        # Learning rate.
        self.alpha = alpha
        if classifier:
            # Number of classes.
            self.k = k
            # One-hot encoded classes.
            self._class = np.empty(self.k)
            # Class-hidden units weights.
            self.weights_c_h = np.random.standard_normal([self.k, self.m]) / np.sqrt(self.k * self.m)
            # Class nodes weights.
            self.bias_c = np.zeros(self.k)
            # Notation comes from 'Neural Networks and Deep Learning' by Charu C. Aggrawal.
            # Current one-hot encoded class.
            self.y = None
            # k x m sigmoid(o_{y,j}) matrix.
            self.sgm_matrix = np.empty([k, m])
            # k x m matrix of exp(o_{y,j}) + 1.
            self.exp_matrix = np.empty([k, m])
            # Vector P(y|x)
            self.p_y_given_x = np.empty(self.k)
            # Sigmoid(y,j)*P(y|x) matrix.
            self.sgm_x_p_matrix = np.empty([k, m])

        return

    def calc_prob_h_given_v(self):
        for j in range(self.m):
            x = self.bias_h[j] + np.dot(self.visible, self.weights[:, j])
            self.prob_h[j] = expit(-x)

        return

    def calc_prob_v_given_h(self):
        for i in range(self.d):
            x = self.bias_v[i] + np.dot(self.hidden, np.transpose(self.weights[i, :]))
            self.prob_v[i] = expit(-x)

        return

    def sample_hidden(self):

        for index, _ in enumerate(self.hidden):
            random_uniform = np.random.uniform()
            if random_uniform < self.prob_h[index]:
                self.hidden[index] = 1
            else:
                self.hidden[index] = 0

        return

    def sample_visible(self):

        for index, _ in enumerate(self.visible):
            random_uniform = np.random.uniform()
            if random_uniform < self.prob_v[index]:
                self.visible[index] = 1
            else:
                self.visible[index] = 0

        return

    def train_cd(self, train_set, batch_size, epochs):
        # Train RBM using a contrastive divergence algorithm.
        # Calculate number of examples in each mini-batch to avoid using if statement in a for loop.
        # In total we have n examples.
        n_examples = len(train_set)
        # That gives n batches.
        n_batches = math.ceil(n_examples / batch_size)
        # Roughly, there are n examples in in each batch.
        n_in_batch = [batch_size] * n_batches
        # With the last batch being possibly different.
        last_batch = n_examples - (n_batches - 1) * batch_size
        # Include possible correction.
        n_in_batch[-1] = last_batch
        # Positive gradient array.
        positive_grad = np.zeros([self.d, self.m])
        # Negative gradient array.
        negative_grad = np.zeros([self.d, self.m])
        # Positive gradient vector for visible units.
        positive_grad_v = np.zeros(self.d)
        # Negative gradient vector for visible units.
        negative_grad_v = np.zeros(self.d)
        # Positive gradient vector for hidden units.
        positive_grad_h = np.zeros(self.m)
        # Negative gradient vector for hidden units.
        negative_grad_h = np.zeros(self.m)
        # Now, back to a core of a CD algorithm.
        # For each epoch.
        for it in range(epochs):
            # Iterate over all mini-batches.
            # last_epoch_weights = self.weights.copy()
            for i in range(n_batches):
                ith_batch_size = n_in_batch[i]
                index_start = sum(n_in_batch[:i])
                positive_grad.fill(0)
                negative_grad.fill(0)
                # Iterate over examples in a batch.
                for j in range(ith_batch_size):
                    index = index_start + j
                    train_example = np.array(train_set[index])
                    self.visible = train_example
                    self.calc_prob_h_given_v()
                    self.sample_hidden()
                    positive_grad += np.dot(train_example.reshape(-1, 1), self.hidden.reshape(1, -1))
                    positive_grad_v += self.visible
                    positive_grad_h += self.hidden
                    self.calc_prob_v_given_h()
                    self.sample_visible()
                    self.calc_prob_h_given_v()
                    self.sample_hidden()
                    negative_grad += np.dot(self.visible.reshape(-1, 1), self.hidden.reshape(1, -1))
                    negative_grad_v += self.visible
                    negative_grad_h += self.hidden
                self.weights -= self.alpha * (positive_grad - negative_grad) / ith_batch_size
                self.bias_v -= self.alpha * (positive_grad_v - negative_grad_v) / ith_batch_size
                self.bias_h -= self.alpha * (positive_grad_h - negative_grad_h) / ith_batch_size
            # print(last_epoch_weights - self.weights)

    # Notation comes from 'Neural Networks and Deep Learning" by Charu C. Aggarwal.

    def calc_sgm_matrix(self):
        o_matrix = self.bias_h + self.weights_c_h + np.dot(self.visible, self.weights)
        self.exp_matrix = np.exp(o_matrix) + 1.
        self.sgm_matrix = expit(o_matrix)

        return

    def calc_prob_y_given_x(self):
        self.calc_sgm_matrix()
        p_denom = np.prod(self.exp_matrix, axis=1).sum()
        for y in range(self.k):
            p_nom = np.prod(self.exp_matrix[y, :])
            self.p_y_given_x[y] = p_nom / p_denom
        self.sgm_x_p_matrix = np.einsum('ij,i->ij', self.sgm_matrix, self.p_y_given_x)

    def update_bias_class(self):
        grad = self._class - self.p_y_given_x
        self.bias_c += self.alpha * grad

    def update_bias_sgd(self):
        grad = self.sgm_x_p_matrix.copy()
        grad[self.y, :] -= self.sgm_matrix[self.y, :]
        self.bias_h -= self.alpha * grad.sum(axis=0)

        return

    def update_c_h_weights(self):
        grad = self.sgm_x_p_matrix.copy()
        grad[self.y, :] -= self.sgm_matrix[self.y, :]
        self.weights_c_h -= self.alpha * grad

        return

    def update_weights(self):
        grad = np.zeros([self.d, self.m, self.k])
        for y in range(self.k):
            grad[:, :, y] = np.dot(self.visible.reshape(-1, 1), self.sgm_x_p_matrix[y, :].reshape(1, -1))
        grad[:, :, self.y] -= np.dot(self.visible.reshape(-1, 1), self.sgm_matrix[self.y, :].reshape(1, -1))
        self.weights -= self.alpha * grad.sum(axis=2)

        return

    def train_sgd(self, train_examples):
        for coding, _class in train_examples:
            self.visible = np.array(coding)
            self._class = np.array(_class)
            self.y = np.argwhere(self._class == 1).item()
            # self.calc_sgm_matrix()
            self.calc_prob_y_given_x()
            self.update_bias_class()
            self.update_bias_sgd()
            self.update_c_h_weights()
            self.update_weights()
            # print(self.weights_c_h)

        return

    def predict(self, feature_vec):
        self.visible = feature_vec
        self.calc_prob_y_given_x()

        return self.p_y_given_x

    def test(self, test_instances):
        error = 0
        for coding, actual in test_instances:
            predicted = self.predict(coding)
            error += np.abs(actual - np.round(predicted)).sum()
        error /= len(test_instances)

        return error
