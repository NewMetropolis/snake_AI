# import rbm
import numpy as np
from a_star import AStarGrid
from snake import SnakeGame
from bfs import BreadthFirstSearchFlat
from graph_stuff import ArticulationPoints

# from graph_stuff import flood_fill, escape_trap
# from a_star import AStarGrid, mark_snakes_way
# from grid_stuff import fill_with_largest_rectangles, largest_area_under_histogram
# import os
# import pickle
# import snake

### Things for testing.
### BFS
# test_gird = np.full([4, 4], fill_value=1)
# # test_gird[:3, 2] = 0
# n_cols = test_gird.shape[1]
# start = 15  # [3, 3]
# end = 8  # [2, 0]
# bfsf = BreadthFirstSearchFlat(test_gird.flatten(), n_cols, start, end)
# end_reachable, track = bfsf.search_sssp()
# bfsf = BreadthFirstSearchFlat(test_gird.flatten(), n_cols, start, end)
# end_reachable, nodes_count = bfsf.search_sssp(return_count=True)
#
# test_gird = np.full([4, 4], fill_value=1)
# snake_ = [15, 14, 10, 6]
# flat_test_grid = test_gird.flatten()
# flat_test_grid[snake_] = 0
# bfsf = BreadthFirstSearchFlat(flat_test_grid, n_cols, start, end, snake_)
# end_reachable, track = bfsf.search_sssp()
# bfsf = BreadthFirstSearchFlat(flat_test_grid, n_cols, start, end, snake_)
# end_reachable, nodes_count = bfsf.search_sssp(return_count=True)
###
### Articulation points
# test_grid = np.full([3, 5], fill_value=1)
# test_grid[0, [2,4]] = 0
# test_grid[2, [0, 1, 2, 4]] = 0
# n_cols = 5
# start = 0
# end = 9
# ap = ArticulationPoints(test_grid.flatten(), n_cols, start, end)
# ap.find(count_nodes=True)
# ap.find_dead_ends()
### A* longest path.
# test_grid = np.full([5, 5], fill_value=1)
# start = [4, 0]
# end = [4, 4]
# test_grid[0, [2, 4]] = 0
# test_grid[1, [2, 4]] = 0
# test_grid[3, 3] = 0
# test_grid[4, 3] = 0
# as_ = AStarGrid(test_grid, start, end)
# as_.compute_longest()
#
# test_grid = np.full([4, 4], fill_value=1)
# start = [3, 2]
# end = [3, 1]
# as_ = AStarGrid(test_grid, start, end)
# as_.compute_longest()

# test_grid = np.full([5, 5], fill_value=1)
# start = [3, 0]
# end = [1, 4]
# as_ = AStarGrid(test_grid, start, end)
# as_.compute_longest()

snake_game = SnakeGame(display_width=400, display_height=440, snake_speed=100, snake_block=20, theme='mark_track',
                       mode='A_star')
snake_game.play_game()
# grid, snake_ = snake_game.astar_game_loop(1)
print('End')


#
# test_grid = np.full([4, 4], fill_value=1)
# # Coordinates should be 2D/consistent with grid.
# snake_ = np.array([[2, 3], [2, 2], [1, 2], [0, 2]])
# test_grid[snake_[:, 0], snake_[:, 1]] = 0
# end = [3, 3]
# asg = AStarGrid(test_grid, snake_[0], end, snake=snake_)
# # track = asg.compute_shortest()
# track = asg.compute_longest()
# print('End.')
# path_='C:\\Users\\Marcin\\PycharmProjects\\snake_nn'
#
# test_grid, _, _, snake_ = pickle.load(open(os.path.join(path_, 'problematic_snake.pkl'), 'rb'))

# colored_grid, n_graphs, reachable_segments = flood_fill(test_grid, snake_[0], snake_)
# escape_trap(colored_grid, snake_, reachable_segments)
# snake_game = snake.SnakeGame(display_width=400, display_height=440, snake_speed=20000, snake_block=20, ai_mode='bfs')
#
# grid, snake_ = snake_game.bfs_game_loop(5)
# colored_grid, n_graphs = flood_fill(grid, (0,0))
# print('End')
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
    """

    :param sequence:
    :param noise:
    :param repeat:
    :return:
    """
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
