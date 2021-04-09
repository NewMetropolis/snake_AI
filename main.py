# import rbm
import bfs
import numpy as np
import pickle
import snake


# snake_game = snake.SnakeGame(display_width=400, display_height=440, snake_speed=50, ai_mode='if_statement')
# bsf_on_a_grid = pickle.load(open('bsf.pkl', 'rb'))
# bsf_on_a_grid.search()
# bsf_on_a_grid.reconstruct_track()
snake_game = snake.SnakeGame(snake_speed=200, ai_mode='bfs')
snake_game.bfs_game_loop(20)
# snake_game.bfs_game_loop(1)
# Running snake game.

# a_grid = np.full([5, 5], 1)
# start = (2, 2)
# end = (2, 4)
# bfs_on_grid = bfs.BreadthFirstSearch(a_grid, start, end)
# bfs_on_grid.search()
# bfs_on_grid.reconstruct_track()
# print(bfs_on_grid.track)


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
#
# with open('DRBM_weights_biases.pkl', 'rb') as pkl_file:
#     weights, weights_c_h, bias_v, bias_h, bias_c = pickle.load(pkl_file)
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
#
#
# snake_ai_game = snake.SnakeGame(ai_mode=True, snake_speed=200)
# rbm_sgd = rbm.RBM(8, 4, alpha=0.1, classifier=True, k=4)
# rbm_sgd.weights = weights
# rbm_sgd.weights_c_h = weights_c_h
# rbm_sgd.bias_v = bias_v
# rbm_sgd.bias_h = bias_h
# rbm_sgd.bias_c = bias_c
# scores, move_coding = snake_ai_game.drbm_game_loop(iterations=3, drbm = rbm_sgd)
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
