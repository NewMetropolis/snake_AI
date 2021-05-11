import numpy as np
import sys
from grid_stuff import reconstruct_track_flatten, validate_move


class BreadthFirstSearchFlat:
    """Breadth First Search on a flattened grid."""

    def __init__(self, grid_flattened, n_columns, start, end, snake=None):
        """Create BreadthFirstSearchFlat instance.

        Args:
            grid_flattened (1D ndarray): Flattened grid to search on.
            n_columns (int): Number of columns in the 2D representation.
            start (int): Node to start at.
            end (int): End node.
            snake (list of int, optional): Snake's segments location.

        Attributes:
            grid (1D ndarray): Flat grid to search on.
            start (int): Node to start at.
            end (int, optional): End node.
            snake (list of int, optional): Snake's segments location.
        """

        self.grid = grid_flattened.copy()
        self.n = self.grid.size
        self.n_cols = n_columns
        self.start = start
        self.end = end
        # Allowed moves on a flattened grid.
        self.moves = [1, -self.n_cols, -1, self.n_cols]
        # For reconstructing paths to nodes.
        self.previous_node = np.full([self.n], fill_value=-1, dtype=int)
        # When there is a "Snake" on the grid its tail will move.
        self.snake = snake

        return

    def search_sssp(self, return_count=False):
        """Search for the single source shortest path.

        Args:
            return_count (bool): If False, return the SSSP between start and end nodes (default). If true return number
            of all nodes reachable from the start.

        Returns:
            end_reachable (bool): Is the end node reachable.
            track (list of int): If 'return_count' is False, return SSSP between start and end nodes.
            nodes_count (int): If 'return_count' is True, return count of all nodes reachable from the start.
        """
        if self.start == self.end:
            # Mark node as visited.
            self.grid[self.end] = 2
            sys.exit('Start and end nodes are the same.')

        reached_end = False
        # For counting number of steps.
        nodes_left_in_layer = 1
        nodes_in_next_layer = 0
        move_count = 0
        queue = [self.start]
        while queue:
            idx = queue.pop(0)
            if self.snake is not None:
                if move_count < len(self.snake):
                    # Check details of the Snake's implementation. It should work that way.
                    segment_to_free = len(self.snake) - move_count - 1
                    self.grid[self.snake[segment_to_free]] = 1
            self.grid[idx] = 2
            # Explore all possible moves.
            for idx_change in self.moves:
                new_idx = idx + idx_change
                if not validate_move(self, idx, idx_change, new_idx):
                    continue
                # if new_idx < 0 or new_idx >= self.n:
                #     continue
                # # Obstacles are marked with 0's, already visited cells with 2's, already enqueued with 3's.
                # elif self.grid[new_idx] != 1:
                #     continue
                # elif idx_change == 1 and new_idx % self.n_cols == 0:
                #     continue
                # elif idx_change == -1 and idx % self.n_cols == 0:
                #     continue
                # Status for enqueued.
                self.grid[new_idx] = 3
                # Keep track of a route.
                self.previous_node[new_idx] = idx
                if new_idx == self.end:
                    reached_end = True
                    if not return_count:
                        self.grid[new_idx] = 2
                        track = reconstruct_track_flatten(self.previous_node, self.start, self.end)
                        return reached_end, track
                # Add to a queue.
                queue.append(new_idx)
                # Keep track of how many nodes are to be visited in next layer.
                nodes_in_next_layer += 1
            nodes_left_in_layer -= 1
            # Did we exit a current layer yet?
            if nodes_left_in_layer == 0:
                nodes_left_in_layer = nodes_in_next_layer
                nodes_in_next_layer = 0
                move_count += 1

        nodes_count = (self.grid == 2).sum() - 1
        return reached_end, nodes_count


class BreadthFirstSearch:
    """Breadth First Search on a grid."""

    def __init__(self, grid, start, end=None, snake_body=None):
        self.grid = grid
        # Number of rows.
        self.n_rows = grid.shape[0]
        # Number of columns.
        self.n_columns = grid.shape[1]
        # Node to start at.
        self.start = start
        # End node.
        self.end = end
        # Allowed row index changes/moves.
        self.r_moves = [0, -1, 0, 1]
        # Allowed column index changes/moves.
        self.c_moves = [1, 0, -1, 0]
        # This will keep track a route.
        self.previous_node = np.full(self.n_rows * self.n_columns, -1, dtype=int)
        # Can be used for navigating Snake in a game.
        self.snake = snake_body
        self.track = []

        return

    def search_sssp(self):
        # A queue for holding a row index.
        r_queue = [self.start[0]]
        # A queue for holding a column index.
        c_queue = [self.start[1]]
        # Avoid duplicates in a queue.
        # enqueued = [0] * self.grid.size
        # For counting number of steps.
        nodes_left_in_layer = 1
        nodes_in_next_layer = 0
        move_count = 0
        while len(r_queue) > 0:
            # Current row index.
            r = r_queue.pop(0)
            # Current column index.
            c = c_queue.pop(0)
            node_index = r * self.n_columns + c
            # Explore all possible moves.
            for ith_direction in range(4):
                r_change = self.r_moves[ith_direction]
                c_change = self.c_moves[ith_direction]
                new_r = r + r_change
                new_c = c + c_change
                new_node_index = new_r * self.n_columns + new_c
                # Still on a grid?
                if new_r < 0 or new_c < 0:
                    continue
                if new_r >= self.n_rows or new_c >= self.n_columns:
                    continue
                # Any obstacles\cell already visited
                if self.grid[new_r, new_c] != 0:
                    continue
                # Keep track of a route.
                self.previous_node[new_node_index] = node_index
                # Have we already reached an end?
                if new_r == self.end[0] and new_c == self.end[1]:
                    # For visual inspection of a grid.
                    self.grid[tuple(self.end)] = -1
                    return self.previous_node
                # Add to a queue.
                r_queue.append(new_r)
                c_queue.append(new_c)
                # Status '2' = visited.
                self.grid[new_r, new_c] = 2
                # Keep track of how many nodes are to be visited in next layer.
                nodes_in_next_layer += 1
            nodes_left_in_layer -= 1
            # Did we exit a current layer yet?
            if nodes_left_in_layer == 0:
                nodes_left_in_layer = nodes_in_next_layer
                nodes_in_next_layer = 0
                move_count += 1
                if self.snake:
                    # Place where the Snake's tail previously was is now empty.
                    if len(self.snake) > 0 and move_count > 1:
                        index_to_empty = tuple(self.snake.pop())
                        self.grid[index_to_empty] = 0

        return

    def reconstruct_track(self):
        # (Re)construct a path given a 'previous node' list from 'bfs_on_grid' function.
        n_columns = self.n_columns
        start = n_columns * self.start[0] + self.start[1]
        end = n_columns * self.end[0] + self.end[1]
        node = end
        if self.previous_node[node] == -1:
            print('No valid path.')

            return

        while node != start:
            row_idx = node // n_columns
            col_idx = node % n_columns
            self.track.append(np.array([row_idx, col_idx]))
            node = self.previous_node[node]
        self.track.reverse()

        return 1