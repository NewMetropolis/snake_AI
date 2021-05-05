import numpy as np
from pqdict import pqdict
from graph_stuff import find_articulation_points, largest_biconnected_component
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
        self.grid_2d = grid_2d
        self.grid = grid_2d.flatten()
        n = grid_2d.size
        self.n = n
        self.n_cols = grid_2d.shape[1]
        self.start = np.ravel_multi_index(start_2d, grid_2d.shape)
        self.end = np.ravel_multi_index(end_2d, grid_2d.shape)
        self.end_2d = end_2d
        self.visited = np.full([n], fill_value=False)
        self.previous = np.full([n], fill_value=-1, dtype=int)
        self.allowed_moves = [1, -self.n_cols, -1, self.n_cols]

    def precompute_snake_distance(self):
        """Precompute Snake/taxicab distance."""

        indexing_2d = np.indices(self.grid_2d.shape)
        row_diff = np.abs(indexing_2d[0] - self.end_2d[0])
        col_diff = np.abs(indexing_2d[1] - self.end_2d[1])
        # noinspection PyAttributeOutsideInit
        self.snake_dist = (row_diff + col_diff).flatten()

    def validate_move(self, node_id, id_change, new_node_id):
        """Check if a move on a grid is valid ."""
        if new_node_id < 0 or new_node_id >= self.n:
            pass
        elif self.grid[new_node_id] == 0:
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
        actual_distance = np.full([self.n], fill_value=np.iinfo(np.int32).max, dtype=int)
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
                valid_move = self.validate_move(node_id, id_change, new_node_id)
                if not valid_move:
                    continue
                # Mind 1, for a grid situation.
                new_distance = actual_distance[node_id] + 1
                if new_distance < actual_distance[new_node_id]:
                    actual_distance[new_node_id] = new_distance
                    indexed_pq[new_node_id] = new_distance + self.snake_dist[new_node_id]
                    self.previous[new_node_id] = node_id
                if new_node_id == self.end:
                    track = reconstruct_track_flatten(self.previous, self.start, self.end)

                    return track

        return

    def longest_path_heuristics(self, node_id):
        nodes_to_visit = []
        for id_change in self.allowed_moves:
            new_node_id = node_id + id_change
            valid_move = self.validate_move(node_id, id_change, new_node_id)
            if valid_move:
                nodes_to_visit.append(new_node_id)
        heuristics =[]
        for node_to_expand in nodes_to_visit:
            for node_to_close in nodes_to_visit:
                if node_to_close != node_to_expand:
                    self.grid[node_to_close] = 0
            # Find articulation points.
            articulation_points = find_articulation_points(self.grid, node_to_expand)
            # Prune the grid.
            not_traversable = largest_biconnected_component(articulation_points, node_to_expand, self.end,
                                                            self.n_cols, mode='prune')

            self.grid[not_traversable] = 0
            # Count the upper limit of the longest path to an end.
            end_reachable, nodes_count = largest_biconnected_component(self.grid, articulation_points,
                                                                       node_to_expand, self.end, self.n_cols,
                                                                       mode='count')
            heuristics.append(nodes_count if end_reachable else 0)
            # Restore the grid.
            self.grid[nodes_to_visit] = 1
            self.grid[not_traversable] = 1

        return nodes_to_visit, heuristics

    def compute_longest(self, snake=None):
        """Compute the longest path.

        Args:
            snake (list: int): Indexes on a flat grid, where the Snake is. When provided, Snake's tail will move as a
            path progresses.

        """
        if self.start == self.end:
            sys.exit('End and start nodes are the same.')
        self.precompute_snake_distance()
        actual_distance = np.full([self.n], fill_value=np.iinfo(np.int32).min, dtype=int)
        actual_distance[self.start] = 0
        h = self.longest_path_heuristics(self.start)
        indexed_pq = pqdict({self.start: actual_distance[self.start] + h})
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
            # Mind 1, for a grid situation.
            new_distance = actual_distance[node_id] + 1
            if new_distance < actual_distance[new_node_id]:
                actual_distance[new_node_id] = new_distance
                indexed_pq[new_node_id] = new_distance + self.snake_dist[new_node_id]
                self.previous[new_node_id] = node_id
            if new_node_id == self.end:
                track = reconstruct_track_flatten(self.previous, self.start, self.end)

                    return track

        return


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
