import numpy as np
from bfs import BreadthFirstSearchFlat
from grid_stuff import validate_move


def find_articulation_points(grid, n_cols, start):
    """Find articulation points on a flat grid using Depth First Search based method.

    Args:
        grid (1D ndarray): Flat grid to search on.
        n_cols (int): Number of columns in the 2D representation.
        start (int): Node to start at.

    Returns:
        articulation_point (lis of int): List of nodes that are articulation points.
    """
    grid = grid.copy()
    n = grid.size
    # Arrays for bookkeeping.
    node_id = np.full([n], fill_value=-1, dtype=int)
    lowpoint = np.full([n], fill_value=np.iinfo(np.int32).max)
    parent_arr = np.full([n], fill_value=-1, dtype=int)
    articulation_points = []
    # Initialize with starting node.
    node_stack = [start]
    backtrack_list = [start]
    parent_arr[start] = start
    current_id = 0
    # This is how we can move on a grid.
    allowed_moves = [1, -n_cols, -1, n_cols]
    while node_stack:
        flat_index = node_stack.pop()
        if grid[flat_index] == 1:
            grid[flat_index] = 2
            node_id[flat_index] = current_id
            lowpoint[flat_index] = current_id
            # lowpoint_for_inspection[np.unravel_index(flat_index, grid.shape)] = lowpoint[flat_index]
            # ids_for_inspection[np.unravel_index(flat_index, grid.shape)] = node_id[flat_index]
            current_id += 1
        for index_change in allowed_moves:
            new_flat_index = flat_index + index_change
            if new_flat_index < 0 or new_flat_index >= n:
                continue
            elif grid[new_flat_index] == 0:
                continue
            elif index_change == 1 and new_flat_index % n_cols == 0:
                continue
            elif index_change == -1 and flat_index % n_cols == 0:
                continue
            elif new_flat_index == parent_arr[flat_index]:
                continue
            if grid[new_flat_index] == 2:
                lowpoint[flat_index] = min(lowpoint[flat_index], node_id[new_flat_index])
                # lowpoint_for_inspection[np.unravel_index(flat_index, grid.shape)] = lowpoint[flat_index]
            else:
                parent_arr[new_flat_index] = flat_index
                node_stack.append(new_flat_index)
                backtrack_list.append(new_flat_index)
    unique, counts = np.unique(parent_arr, return_counts=True)
    child_count = dict(zip(unique, counts))
    child_count[start] -= 1
    while backtrack_list:
        child = backtrack_list.pop()
        parent = parent_arr[child]
        lowpoint[parent] = min(lowpoint[parent], lowpoint[child])
        # lowpoint_for_inspection[np.unravel_index(parent, grid.shape)] = lowpoint[parent]
        if node_id[parent] <= lowpoint[child]:
            articulation_points.append(parent)
            if parent == start and child_count[start] < 2:
                articulation_points.pop()

    return set(articulation_points)


def find_dead_ends(grid, n_cols, articulation_points, start, end, snake=None):
    """Find nodes that can be traversed on a simple path between start and end."""
    n = grid.size
    allowed_moves = [1, -n_cols, -1, n_cols]
    not_traversable = []
    for idx in articulation_points:
        grid_copy = grid.copy()
        grid_copy[idx] = 0
        for idx_change in allowed_moves:
            new_idx = idx + idx_change
            if new_idx < 0 or new_idx >= n:
                continue
            elif grid_copy[new_idx] != 1:
                continue
            elif idx_change == 1 and new_idx % n_cols == 0:
                continue
            elif idx_change == -1 and idx % n_cols == 0:
                continue
            if new_idx == end or new_idx == start:
                continue
            # There should be 'reset object' function or so, probably.

            bfs = BreadthFirstSearchFlat(grid_copy, n_cols, new_idx, end, snake)
            end_reachable, _ = bfs.search_sssp()
            if not end_reachable:
                bfs = BreadthFirstSearchFlat(grid_copy, n_cols, new_idx, start, snake)
                start_reachable, track_to_start = bfs.search_sssp()
                if not start_reachable:
                    not_traversable = not_traversable + list(np.argwhere(bfs.grid == 2).flatten())

    return not_traversable

