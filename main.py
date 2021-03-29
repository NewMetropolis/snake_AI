import pickle
import snake
import rbm
import numpy as np


def bfs_on_grid(grid, start, end):
    """Breadth First Search on a grid. Returns a minimal number of steps between start and end points."""
    # Number of rows.
    n_rows = grid.shape[0]
    # Number of columns.
    n_columns = grid.shape[1]
    # A queue for holding a row index.
    r_queue = [start[0]]
    # A queue for holding a column index.
    c_queue = [start[1]]
    # Allowed row index changes/moves.
    r_moves = [0, -1, 0, 1]
    # Allowed column index changes/moves.
    c_moves = [1, 0, -1, 0]
    # This will track a route.
    previous_node = np.full(n_rows * n_columns, -1, dtype=int)

    while len(r_queue) > 0:
        # Current row index.
        r = r_queue.pop()
        # Current column index.
        c = c_queue.pop()
        node_index = r * n_columns + c
        # Mark as visited/now unreachable.
        grid[r, c] = 0
        # Explore all possible moves.
        for ith_direction in range(4):
            r_change = r_moves[ith_direction]
            c_change = c_moves[ith_direction]
            new_r = r + r_change
            new_c = c + c_change
            new_node_index = new_r * n_columns + new_c
            # Still on a grid?
            if new_r < 0 or new_c < 0:
                continue
            if new_r >= n_rows or new_c >= n_columns:
                continue
            # Any obstacles (marked as 0s)?
            if grid[new_r, new_c] == 0:
                continue
            # Keep track of a route.
            previous_node[new_node_index] = node_index
            # Have we already reached an end?
            if new_r == end[0] and new_c == end[1]:
                return previous_node
            # Add to a queue.
            r_queue.append(new_r)
            c_queue.append(new_c)

    return previous_node

grid = np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]])
route = bfs_on_grid(grid, (0, 0), (3, 2))
print('Hi')

def add_noise(sequence, noise, repeat):
    # Function adds noise to a binary sequence of an arbitrary length.
    n_bits = len(sequence)
    n_perturbed = int(n_bits * noise)
    sequence_list = []

    for seq in range(repeat):
        pert_sequence = sequence.copy()
        for nth in range(n_perturbed):
            bit_n = np.random.randint(0, n_bits)
            pert_sequence[bit_n] ^= 1
        sequence_list.append(pert_sequence)

    return sequence_list



#
# output_list = []
#
#
# def test():
#     for i in range(10):
#         rbm = RBM(5, 1, 0.1)
#         # add_noise([0, 0, 0, 0, 0], .2, 10)
#         # train_examples = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
#         # train_examples = [[1,1,0,0], [0,1,1,0], [1,0,0,1]]
#         train_examples = add_noise([0, 0, 0, 0, 0], .2, 10)
#         rbm.train_cd(train_examples, 1, 100)
#         rbm.visible = [1, 1, 1, 1, 1]
#         for j in range(1000):
#             rbm.calc_prob_h_given_v()
#             # print(rbm.prob_h)
#             rbm.sample_hidden()
#             rbm.calc_prob_v_given_h()
#             # print(rbm.prob_v)
#             rbm.sample_visible()
#             output_list.append(rbm.visible)
#             # print(rbm.visible)


# def average_output(output_list):
#     output_arr = np.array(output_list, dtype=int)
#     return np.mean(output_arr, axis=0)

with open('DRBM_weights_biases.pkl', 'rb') as pkl_file:
    weights, weights_c_h, bias_v, bias_h, bias_c = pickle.load(pkl_file)
# test()
# print(average_output(output_list))
#
# move_coding = np.load('moves_coding_ai_bool.npy')
# unique_move_coding = np.unique(move_coding, axis=0)
# snake_coding = []
# for n in range(move_coding.shape[0]):
#     snake_feature = move_coding[n, :8]
#     snake_class = move_coding[n, 8:]
#     snake_coding.append((snake_feature, snake_class))
# rbm_sgd = rbm.RBM(8, 4, alpha=0.1, classifier=True, k=4)
# # train = [([1, 1, 1], [0, 1]), ([0, 0, 0], [1, 0]), ([1, 1, 0], [0, 1]), ([1, 0, 0], [0, 1])]
# for epoch in range(5):
#     rbm_sgd.train_sgd(snake_coding)
# print(rbm_sgd.test(snake_coding))
# with open('DRBM_weights_biases.pkl', 'wb') as pkl_file:
#     pickle.dump((rbm_sgd.weights, rbm_sgd.weights_c_h, rbm_sgd.bias_v, rbm_sgd.bias_h, rbm_sgd.bias_c), pkl_file)
print('HI')

# Running snake game.

snake_ai_game = snake.SnakeGame(ai_mode=True, snake_speed=200)
rbm_sgd = rbm.RBM(8, 4, alpha=0.1, classifier=True, k=4)
rbm_sgd.weights = weights
rbm_sgd.weights_c_h = weights_c_h
rbm_sgd.bias_v = bias_v
rbm_sgd.bias_h = bias_h
rbm_sgd.bias_c = bias_c
scores, move_coding = snake_ai_game.drbm_game_loop(iterations=3, drbm = rbm_sgd)
# snake_game = SnakeGame()
# snake_game.game_loop()

# Load moves coding.


# Hopfield network (to be reviewed).

# network = HopfieldNetwork(8)
# network.train(unique_move_coding, mode='ith', start=0, stop=4, skip=4)
# # network.train(unique_move_coding, mode='all_edges', start=4, stop=8)
# # network.train(unique_move_coding, mode='one_to_all', start=4, stop=8, start_2=8, stop_2=12)
# # network.recall([0,1,1,1,0,0,0,0,1,0,0,0,0], mode='move')
# # toy_example = [[1, 0, 0, 1], [0, 1, 1, 0]]
# # network.train_on_all(np.array(np.array(toy_example).reshape(-1,4)))
# # network.recall([1, 0, 0, 0])
