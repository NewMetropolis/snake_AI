import numpy as np


def flood_fill(grid, start, snake=None):
    """Flood fill on a grid.
    Returns labeled grid and a total number of disjoint graphs."""

    colored_grid = grid.copy()
    n_rows = grid.shape[0]
    n_cols = grid.shape[1]
    # Allowed row index changes/moves.
    r_moves = [0, -1, 0, 1]
    # Allowed column index changes/moves.
    c_moves = [1, 0, -1, 0]
    # Keep track of which nodes have been already visited and which already enqueued.
    # visited = np.full( grid.shape, fill_value=False)
    # enqueued = np.full( grid.shape, fill_value=False)
    # Row index queue.
    r_queue = [start[0]]
    # Column index queue.
    c_queue = [start[1]]
    # Avoid duplicates in a queue.
    # For counting number of steps.
    nodes_left_in_layer = 1
    nodes_in_next_layer = 0
    move_count = 0
    current_label = 10
    if snake:
        reachable = np.full(len(snake), fill_value=False)
    while 1:
        while len(r_queue) > 0:
            # Current row index.
            r = r_queue.pop(0)
            # Current column index.
            c = c_queue.pop(0)
            # Mark as visited.
            colored_grid[r, c] = current_label
            # Explore all possible moves.
            for ith_direction in range(4):
                r_change = r_moves[ith_direction]
                c_change = c_moves[ith_direction]
                new_r = r + r_change
                new_c = c + c_change
                # new_node_index = new_r * n_columns + new_c
                # Still on a grid?
                if new_r < 0 or new_c < 0:
                    continue
                if new_r >= n_rows or new_c >= n_cols:
                    continue
                # Any obstacles\cell already visited?
                if colored_grid[new_r, new_c] != 0:
                    if snake:
                        if current_label == 10 and colored_grid[new_r, new_c] == 1:
                            snake_segment_idx = np.argwhere((np.array(snake) == [new_r, new_c]).all(axis=1))
                            if snake_segment_idx.size == 1:
                                reachable[snake_segment_idx.item()] = True
                    continue
                # # 2 is a status for 'enqueued'.
                # if colored_grid[new_r, new_c] == 2:
                #     continue
                # Add to a queue.
                r_queue.append(new_r)
                c_queue.append(new_c)
                colored_grid[new_r, new_c] = 2
                # Keep track of how many nodes are to be visited in next layer.
                nodes_in_next_layer += 1
            nodes_left_in_layer -= 1
            # Did we exit a current layer yet?
            if nodes_left_in_layer == 0:
                nodes_left_in_layer = nodes_in_next_layer
                nodes_in_next_layer = 0

        current_label += 1
        try:
            new_start_node = tuple(np.argwhere(colored_grid == 0)[0])
            r_queue.append(new_start_node[0])
            c_queue.append(new_start_node[1])
        except IndexError:
            if snake:
                return colored_grid, current_label - 10, reachable
            else:
                return colored_grid, current_label - 10

    return


def escape_trap(colored_grid, snake, reachable):
    """When the Snake is self-trapped, check if an escape route exists."""

    # Check an upper bound condition.
    unique_labels, labels_count = np.unique(colored_grid, return_counts=True)
    graphs_size = dict(zip(unique_labels, labels_count))
    last_reachable_segment = np.argwhere(reachable)[-1][0]
    # -1 ?
    dist_from_tail = len(snake) - last_reachable_segment
    if dist_from_tail > graphs_size[10]:
        print('No way to ecape.')
        return

    n_rows = colored_grid.shape[0]
    n_cols = colored_grid.shape[1]
    # Allowed row index changes/moves.
    r_moves = [0, -1, 0, 1]
    # Allowed column index changes/moves.
    c_moves = [1, 0, -1, 0]
    explored_directions_l = [0]
    # Row index queue.
    r_queue = [snake[0][0]]
    # Column index queue.
    c_queue = [snake[0][1]]
    longest_paths = [[]] * len(snake)
    shortest_paths = [[]] * len(snake)
    optimal_paths = [[]] * len(snake)
    longest_paths_length = [-np.inf] * len(snake)
    shortest_paths_length = [np.inf] * len(snake)
    while len(r_queue) > 0:
        explored_directions = explored_directions_l[-1]
        if explored_directions == 4:
            r_queue.pop()
            c_queue.pop()
            explored_directions_l.pop()
        else:
            # Current path length.
            path_length = len(r_queue)
            # Current row index.
            r = r_queue[-1]
            # Current column index.
            c = c_queue[-1]
            r_change = r_moves[explored_directions]
            c_change = c_moves[explored_directions]
            explored_directions_l[-1] += 1
            new_r = r + r_change
            new_c = c + c_change
            # Still on a grid?
            if new_r < 0 or new_c < 0:
                continue
            if new_r >= n_rows or new_c >= n_cols:
                continue
            node_enlisted = np.argwhere((np.array(list(zip(r_queue, c_queue))) == [new_r, new_c]).all(axis=1))
            if node_enlisted.size == 1:
                continue
            #
            if colored_grid[new_r, new_c] == 1:
                snake_segment_idx = np.argwhere((np.array(snake) == [new_r, new_c]).all(axis=1))
                if snake_segment_idx.size == 1:
                    snake_segment_idx = snake_segment_idx.item()
                    if path_length > longest_paths_length[snake_segment_idx]:
                        longest_paths_length[snake_segment_idx] = path_length
                        longest_paths[snake_segment_idx] = list(zip(r_queue, c_queue))
                    # What about paths of equal length?
                    elif path_length < shortest_paths_length[snake_segment_idx]:
                        shortest_paths_length[snake_segment_idx] = path_length
                        shortest_paths[snake_segment_idx] = list(zip(r_queue, c_queue))
                    # Chyba jest ok.
                    dist_from_tail = len(snake) - snake_segment_idx
                    # Same question.
                    if path_length == dist_from_tail:
                        optimal_paths[snake_segment_idx] = list(zip(r_queue, c_queue))
            else:
                r_queue.append(new_r)
                c_queue.append(new_c)
                explored_directions_l.append(0)

    return
