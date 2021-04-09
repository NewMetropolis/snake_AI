import numpy as np

class BreadthFirstSearch:
    """Breadth First Search on a grid."""

    def __init__(self, grid, start, end, snake_body=None):
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

    def search(self):
        # A queue for holding a row index.
        r_queue = [self.start[0]]
        # A queue for holding a column index.
        c_queue = [self.start[1]]
        # Avoid duplicates in a queue.
        enqueued = [0] * self.grid.size
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
            # Mark as visited/now unreachable.
            self.grid[r, c] = 0
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
                # Any obstacles (marked as 0s)?
                if self.grid[new_r, new_c] == 0:
                    continue
                # Keep track of a route.
                self.previous_node[new_node_index] = node_index
                # Have we already reached an end?
                if new_r == self.end[0] and new_c == self.end[1]:
                    return self.previous_node
                if enqueued[new_node_index] == 1:
                    continue
                # Add to a queue.
                r_queue.append(new_r)
                c_queue.append(new_c)
                enqueued[new_node_index] = 1
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
                        self.grid[index_to_empty] = 1

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
    