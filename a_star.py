import numpy as np
from pqdict import pqdict
import sys

"""Implementation of A* algorithm"""


class AStarGrid:
    """A^{*} algorithm to compute either longest or shortest path on a grid

    Attributes:
        grid (1D ndarray): Flattened grid_2D.
        n (int): Total grid size.
        n_cols (int): Number of columns.
        start (int): Start node index on the flat grid.
        end (int): End node index on the flat grid.
    """

    def __init__(self, grid_2d, start_2d, end_2d):
        """Initialize AStarGrid object.

        Args:
            grid_2d (2D ndarray): A grid to find path on. '1' marks traversable nodes, '0' otherwise.
            start_2d (Union[1D list,tuple, 1D array]): Coordinates [row number, column number] for a start node.
            end_2d (Union[1D list,tuple, 1D array]): Coordinates [row number, column number] for an end node.
        """
        # grid, start, end
        # grid_for_inspection = grid.copy()
        self.grid = grid_2d.flatten()
        n = grid.size
        self.n = n
        self.n_cols = grid.shape[1]
        self.start = np.ravel_multi_index(start_2d, n)
        self.end = np.ravel_multi_index(end_2d, n)
        self.visited = np.full([n], fill_value=False)
        self.previous = np.full([n], fill_value=-1, dtype=int)
        self.allowed_moves = [1, -self.n_cols, -1, self.n_cols]

    def precompute_snake_distance(self):
        """Precompute Snake/taxicab distance."""
        indexing_2d = np.indices(grid.shape)
        row_diff = np.abs(indexing_2d[0] - end[0])
        col_diff = np.abs(indexing_2d[1] - end[1])
        # noinspection PyAttributeOutsideInit
        self.snake_dist = (row_diff + col_diff).flatten()

    def validate_move(self, node_id, id_change, new_node_id):
        """Check if a move on a grid is valid ."""
        if new_node_id < 0 or new_node_id >= self.n:
            pass
        elif grid[new_node_id] == 0:
            pass
        elif self.visited[new_node_id]:
            pass
        elif id_change == 1 and new_node_id % self.n_cols == 0:
            pass
        elif id_change == -1 and node_id % self.n_cols == 0:
            pass
        else:
            return 1

    def compute_shortest(self, snake=None):
        """Compute the shortest path.

        Args:
            snake (list: int): Indexes on a flat grid, where the Snake is. When provided, Snake's tail will move as a
            path progresses.

        """
        if self.start == self.end:
            sys.exit('End and start nodes are the same.')
        self.precompute_snake_distance()
        actual_distance = np.full([self.n], fill_value=np.inf)
        actual_distance[self.start] = 0
        indexed_pq = pqdict({self.start: actual_distance[self.start] + self.snake_dist[self.start]})
        nodes_to_free = []
        while indexed_pq:
            if snake:
                self.grid[nodes_to_free] = 0
            node_id = indexed_pq.pop()
            if snake:
                if actual_distance[node_id] >= len(snake):
                    nodes_to_free = snake
                else:
                    # Check details of the Snake's implementation. It should work that way.
                    nodes_to_free = snake[-(actual_distance[node_id] + 1):]
                self.grid[nodes_to_free] = 1
            # position_2d = np.unravel_index(node_id, grid.shape)
            # grid_for_inspection[position_2d] = -3
            self.visited[node_id] = True
            for id_change in self.allowed_moves:
                new_node_id = node_id + id_change
                self.validate_move(node_id, id_change, new_node_id)
                # Mind 1, for a grid situation.
                new_distance = actual_distance[node_id] + 1
                if new_distance < actual_distance[new_node_id]:
                    actual_distance[new_node_id] = new_distance
                    indexed_pq[new_node_id] = new_distance + self.snake_dist[new_node_id]
                    self.previous[new_node_id] = node_id
                if new_node_id == self.end:
                    track = reconstruct_track_flatten(self.previous, self.start, self.end)

                    return track


def reconstruct_track_flatten(previous, start, end):
    """Reconstruct a track on a flattened grid."""
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
