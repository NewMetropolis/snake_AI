import numpy as np
from pqdict import pqdict
"""Implementation of A* algorithm"""


def a_star_grid(grid, start, end):
    """A* algorithm on a grid."""
    grid_for_inspection = grid.copy()
    n_cols = grid.shape[1]
    n = grid.size
    visited = np.full([n], fill_value=False)
    previous = np.full([n], fill_value=-1, dtype=int)
    actual_distance = np.full([n], fill_value=np.inf)
    flatten = grid.flatten()
    start_f = np.ravel_multi_index(start, grid.shape)
    end_f = np.ravel_multi_index(end, grid.shape)
    allowed_moves = [1, -n_cols, -1, n_cols]
    # Precompute heuristics.
    indexing_2d = np.indices(grid.shape)
    row_diff = np.abs(indexing_2d[0] - end[0])
    col_diff = np.abs(indexing_2d[1] - end[1])
    snake_dist = (row_diff + col_diff).flatten()
    actual_distance[start_f] = 0
    indexed_pq = pqdict({start_f: actual_distance[start_f] + snake_dist[start_f]})

    while indexed_pq:
        node_id = indexed_pq.pop()
        position_2d = np.unravel_index(node_id, grid.shape)
        grid_for_inspection[position_2d] = -3
        visited[node_id] = True
        for id_change in allowed_moves:
            new_node_id = node_id + id_change
            if new_node_id < 0 or new_node_id >= n:
                continue
            elif flatten[new_node_id] == 0:
                continue
            elif visited[new_node_id]:
                continue
            elif id_change == 1 and new_node_id % n_cols == 0:
                continue
            elif id_change == -1 and node_id % n_cols == 0:
                continue
            # Mind 1, for a grid situation.
            new_distance = actual_distance[node_id] + 1
            if new_distance < actual_distance[new_node_id]:
                actual_distance[new_node_id] = new_distance
                indexed_pq[new_node_id] = new_distance + snake_dist[new_node_id]
                previous[new_node_id] = node_id
            if new_node_id == end_f:
                track = reconstruct_track_flatten(previous, start_f, end_f)

                return track

    return


# This is very similar to one of BreadthFirstSearch class methods.
def reconstruct_track_flatten(previous, start, end):
    track = []
    node = end
    if previous[node] == -1:
        print('No valid path.')
        return

    while node != start:
        track.append(node)
        node = previous[node]
    track.reverse()

    return track


def mark_snakes_way(grid, track):
    indexes_2d = np.unravel_index(track, grid.shape)
    for i in range(indexes_2d[0].size):
        row_idx = indexes_2d[0][i]
        col_idx = indexes_2d[1][i]
        grid[row_idx, col_idx] = -2

    return