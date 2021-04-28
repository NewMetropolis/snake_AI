import numpy as np
from pqdict import pqdict
"""Implementation of A* algorithm"""


def a_star_grid(grid, start, end):
    """A* algorithm on a grid."""
    n_rows = grid.shape[0]
    n_cols = grid.shape[1]
    n = grid.size
    visited = np.full([n], fill_value=False)
    previous = np.full([n], fill_value=-1, dtype=int)
    min_distance = np.full([n], fill_value=np.inf)
    flatten = grid.flatten()
    start_f = np.ravel_multi_index(start, grid.shape)
    end_f = np.ravel_multi_index(end, grid.shape)
    allowed_moves = [1, -n_cols, -1, n_cols]
    indexed_pq = pqdict({start_f: 0})

    def heuristics(flattened_id):
        # A.k.a 'snake distance'.
        node_idx = np.array(np.unravel_index(flattened_id, grid.size))
        abs_diff = np.abs(np.subtract(end, node_idx))
        return 1/abs_diff.sum()

    min_distance[start_f] = 0 + heuristics(start_f)

    while indexed_pq:
        node_id, distance = indexed_pq.popitem()
        visited[node_id] = True
        # This probably will not happen with IPQ. Can it?
        if min_distance[node_id] < distance:
            print("A jednak.")
            continue
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
            new_distance = min_distance[node_id] + 0 + heuristics(new_node_id)
            if new_distance < min_distance[new_node_id]:
                min_distance[new_node_id] = new_distance
                indexed_pq[new_node_id] = new_distance
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
