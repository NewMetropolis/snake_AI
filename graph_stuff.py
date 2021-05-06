from bfs import BreadthFirstSearchFlat
import numpy as np


def find_articulation_points(grid, start):
    """Find articulation points on a grid using Depth First Search based method."""
    # For debugging.
    ids_for_inspection = grid.copy()
    lowpoint_for_inspection = grid.copy()
    # Grid related values.
    flatten = grid.flatten()
    start_f = np.ravel_multi_index(start, grid.shape)
    n_cols = grid.shape[1]
    n = grid.size
    # Arrays for an algorithm bookkeeping.
    visited = np.full([n], fill_value=False)
    node_id = np.full([n], fill_value=-1, dtype=int)
    lowpoint = np.full([n], fill_value=np.inf)
    parent_arr = np.full([n], fill_value=-1, dtype=int)
    articulation_point = np.full([n], fill_value=False)
    # Initialize with starting node.
    node_stack = [start_f]
    backtrack_list = [start_f]
    parent_arr[start_f] = start_f
    current_id = 0
    # This is how we can move on a grid.
    allowed_moves = [1, -n_cols, -1, n_cols]
    n_moves = 1
    last_node = start_f
    while node_stack:
        flat_index = node_stack.pop()
        if not visited[flat_index]:
            visited[flat_index] = True
            node_id[flat_index] = current_id
            lowpoint[flat_index] = current_id
            lowpoint_for_inspection[np.unravel_index(flat_index, grid.shape)] = lowpoint[flat_index]
            ids_for_inspection[np.unravel_index(flat_index, grid.shape)] = node_id[flat_index]
            current_id += 1
            if parent_arr[flat_index] == last_node:
                n_moves += 1
        if parent_arr[last_node] == flat_index:
            n_moves -= 1
        for index_change in allowed_moves:
            new_flat_index = flat_index + index_change

            if new_flat_index < 0 or new_flat_index >= n:
                continue
            elif flatten[new_flat_index] == 0:
                continue
            elif index_change == 1 and new_flat_index % n_cols == 0:
                continue
            elif index_change == -1 and flat_index % n_cols == 0:
                continue
            elif new_flat_index == parent_arr[flat_index]:
                continue
            if visited[new_flat_index]:
                lowpoint[flat_index] = min(lowpoint[flat_index], node_id[new_flat_index])
                lowpoint_for_inspection[np.unravel_index(flat_index, grid.shape)] = lowpoint[flat_index]
            else:
                parent_arr[new_flat_index] = flat_index
                node_stack.append(new_flat_index)
                backtrack_list.append(new_flat_index)
        last_node = flat_index
    while backtrack_list:
        child = backtrack_list.pop()
        parent = parent_arr[child]
        lowpoint[parent] = min(lowpoint[parent], lowpoint[child])
        lowpoint_for_inspection[np.unravel_index(parent, grid.shape)] = lowpoint[parent]
        if node_id[parent] <= lowpoint[child]:
            articulation_point[parent] = True
    unique, counts = np.unique(parent_arr, return_counts=True)
    child_count = dict(zip(unique, counts))
    articulation_point[start_f] = child_count[start_f] > 2

    return articulation_point


def largest_biconnected_component(grid_flattened, articulation_points, start, end, n_cols):
    """Finds largest biconnected components/traversable simple paths between start and end."""
    n = grid_flattened.size
    allowed_moves = [1, -n_cols, -1, n_cols]
    articulation_indexes = np.array(np.arange(n))[articulation_points]
    not_traversable = []
    for idx in articulation_indexes:
        flattened_copy = grid_flattened.copy()
        flattened_copy[idx] = 0

        for idx_change in allowed_moves:
            new_idx = idx + idx_change
            if new_idx < 0 or new_idx >= n:
                continue
            elif flattened_copy[new_idx] == 0:
                continue
            elif idx_change == 1 and new_idx % n_cols == 0:
                continue
            elif idx_change == -1 and idx % n_cols == 0:
                continue
            # There should be 'reset object' function or so, probably.
            bfs = BreadthFirstSearchFlat(flattened_copy, n_cols, new_idx, end)
            end_reachable, _ = bfs.search_sssp()
            if not end_reachable:
                bfs = BreadthFirstSearchFlat(flattened_copy, n_cols, new_idx, start)
                start_reachable, track_to_start = bfs.search_sssp()
                if not start_reachable:
                    not_traversable = not_traversable + list(np.argwhere(bfs.grid == 2).flatten())
                    grid_flattened[bfs.grid == 2] = 0

    return not_traversable


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
    # -1 ? Looks ok
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
    optimal_paths_length = [np.inf] * len(snake)
    # The thing with head. Escape where the head was?
    n_paths = 0
    while len(r_queue) > 0:
        explored_directions = explored_directions_l[-1]
        if explored_directions == 4:
            r_queue.pop()
            c_queue.pop()
            explored_directions_l.pop()
            n_paths += 1
            if n_paths % 1000 == 0:
                print(n_paths)
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
                    # What about paths of equal lengths?
                    if path_length < shortest_paths_length[snake_segment_idx]:
                        shortest_paths_length[snake_segment_idx] = path_length
                        shortest_paths[snake_segment_idx] = list(zip(r_queue, c_queue))
                    # Chyba jest ok.
                    dist_from_tail = len(snake) - snake_segment_idx
                    # Same question.
                    if dist_from_tail <= path_length <= optimal_paths_length[snake_segment_idx]:
                        optimal_paths_length[snake_segment_idx] = path_length
                        optimal_paths[snake_segment_idx] = list(zip(r_queue, c_queue))

            else:
                r_queue.append(new_r)
                c_queue.append(new_c)
                explored_directions_l.append(0)
    print(n_paths)
    return


def escape_trap_2(colored_grid, snake, reachable):
    """When the Snake is self-trapped, check if an escape route exists."""

    # Check an upper bound condition.
    unique_labels, labels_count = np.unique(colored_grid, return_counts=True)
    graphs_size = dict(zip(unique_labels, labels_count))
    last_reachable_segment = np.argwhere(reachable)[-1][0]
    # -1 ? Looks ok
    dist_from_tail = len(snake) - last_reachable_segment
    if dist_from_tail > graphs_size[10]:
        print('No way to escape.')
        return

    n_rows = colored_grid.shape[0]
    n_cols = colored_grid.shape[1]
    # Allowed row index changes/moves.
    r_moves = [0, -1, 0, 1]
    # Allowed column index changes/moves.
    c_moves = [1, 0, -1, 0]
    # Row index queue.
    paths_queue = [tuple(snake[0])]
    longest_paths = [[]] * len(snake)
    shortest_paths = [[]] * len(snake)
    optimal_paths = [[]] * len(snake)
    longest_paths_length = [-np.inf] * len(snake)
    shortest_paths_length = [np.inf] * len(snake)
    optimal_paths_length = [np.inf] * len(snake)
    # The thing with head. Escape where the head was?
    n_paths = 0
    while len(paths_queue) > 0:
            current_path = paths_queue.pop()
            if len(paths_queue) % 1000 == 0:
                print(len(paths_queue))
    for ith_direction in range(4):
            # Current path length.
            path_length = len(current_path)
            # Last node row index.
            r = current_path[-1][0]
            # Current column index.
            c = current_path[-1][1]
            r_change = r_moves[ith_direction]
            c_change = c_moves[ith_direction]
            new_r = r + r_change
            new_c = c + c_change
            # Still on a grid?
            if new_r < 0 or new_c < 0:
                continue
            if new_r >= n_rows or new_c >= n_cols:
                continue
            node_enlisted = np.argwhere((np.array(current_path) == [new_r, new_c]).all(axis=1))
            if node_enlisted.size == 1:
                continue
            paths_queue.append(current_path + [(new_r, new_c)])
            if colored_grid[new_r, new_c] == 1:
                snake_segment_idx = np.argwhere((np.array(snake) == [new_r, new_c]).all(axis=1))
                if snake_segment_idx.size == 1:
                    snake_segment_idx = snake_segment_idx.item()
                    if path_length > longest_paths_length[snake_segment_idx]:
                        longest_paths_length[snake_segment_idx] = path_length
                        longest_paths[snake_segment_idx] = current_path
                    # What about paths of equal lengths?
                    if path_length < shortest_paths_length[snake_segment_idx]:
                        shortest_paths_length[snake_segment_idx] = path_length
                        shortest_paths[snake_segment_idx] = current_path
                    # Chyba jest ok.
                    dist_from_tail = len(snake) - snake_segment_idx
                    # Same question.
                    if dist_from_tail <= path_length <= optimal_paths_length[snake_segment_idx]:
                        optimal_paths_length[snake_segment_idx] = path_length
                        optimal_paths[snake_segment_idx] = current_path


    print(len(paths_queue))
    return



