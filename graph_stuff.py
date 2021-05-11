import numpy as np
from bfs import BreadthFirstSearchFlat
from grid_stuff import validate_move


class ArticulationPoints:
    """Mark articulation points and 'dead ends' on a grid graph using Depth First Search based method.

       Attributes:
            grid (1D ndarray): Flat grid to search on.
            n_cols (int): Number of columns in the 2D representation.
            start (int): Node to start at.
            articulation_points (list of int): Indexes of nodes that are articulation points.
            end (int, optional): Index of an end node. If provided function will
            end_reachable (bool, optional): If an end node index is provided 'find' method will check whether it is
            reachable.
            nodes_count (int, optional): Number of all reachable nodes. See 'count_nodes' in the 'find'
            method.
    """

    def __init__(self, grid, n_cols, start, end=None):
        """Initialize ArticulationPoints object.

        Args:
            grid (1D ndarray): Flat grid to search on.
            n_cols (int): Number of columns in the 2D representation.
            start (int): Node to start at.
            end (int, optional): Index of an end node. If provided function will

        """
        # Doczytać gdzie powinien być atrybut traversable nodes.
        self.original_grid = grid
        self.grid = grid.copy()
        self.n = grid.size
        self.n_cols = n_cols
        self.start = start
        self.end = end
        self.articulation_points = []
        # This is how we can move on a grid.
        self.allowed_moves = [1, -n_cols, -1, n_cols]
        self.nodes_count = None
        self.end_reachable = None
        self.not_traversable = None

    def find(self, count_nodes=False):
        """Find articulation points on a flat grid.
            Args:
                count_nodes (bool, optional): If True return count of all accessible nodes. Default is False.
        """
        # Arrays for bookkeeping.
        node_id = np.full([self.n], fill_value=-1, dtype=int)
        lowpoint = np.full([self.n], fill_value=np.iinfo(np.int32).max)
        parent_arr = np.full([self.n], fill_value=-1, dtype=int)
        # Initialize with starting node.
        node_stack = [self.start]
        backtrack_list = [self.start]
        parent_arr[self.start] = self.start
        current_id = 0
        while node_stack:
            flat_index = node_stack.pop()
            if self.grid[flat_index] == 1:
                self.grid[flat_index] = 2
                node_id[flat_index] = current_id
                lowpoint[flat_index] = current_id
                # lowpoint_for_inspection[np.unravel_index(flat_index, grid.shape)] = lowpoint[flat_index]
                # ids_for_inspection[np.unravel_index(flat_index, grid.shape)] = node_id[flat_index]
                current_id += 1
            for index_change in self.allowed_moves:
                new_flat_index = flat_index + index_change
                if new_flat_index < 0 or new_flat_index >= self.n:
                    continue
                elif self.grid[new_flat_index] == 0:
                    continue
                elif index_change == 1 and new_flat_index % self.n_cols == 0:
                    continue
                elif index_change == -1 and flat_index % self.n_cols == 0:
                    continue
                elif new_flat_index == parent_arr[flat_index]:
                    continue
                if self.grid[new_flat_index] == 2:
                    lowpoint[flat_index] = min(lowpoint[flat_index], node_id[new_flat_index])
                    # lowpoint_for_inspection[np.unravel_index(flat_index, grid.shape)] = lowpoint[flat_index]
                else:
                    parent_arr[new_flat_index] = flat_index
                    node_stack.append(new_flat_index)
                    backtrack_list.append(new_flat_index)
        unique, counts = np.unique(parent_arr, return_counts=True)
        child_count = dict(zip(unique, counts))
        child_count[self.start] -= 1
        if count_nodes:
            self.nodes_count = len(set(backtrack_list)) - 1
        if self.end:
            self.end_reachable = self.end in backtrack_list
        while backtrack_list:
            child = backtrack_list.pop()
            parent = parent_arr[child]
            lowpoint[parent] = min(lowpoint[parent], lowpoint[child])
            # lowpoint_for_inspection[np.unravel_index(parent, grid.shape)] = lowpoint[parent]
            if node_id[parent] <= lowpoint[child]:
                self.articulation_points.append(parent)
                if parent == self.start and child_count[self.start] < 2:
                    self.articulation_points.pop()
        self.articulation_points = set(self.articulation_points)

        return

    def find_dead_ends(self):
        """Find nodes that can not be traversed on a simple path from a start to an end.

            Returns:
                not_traversable (set of int): Indexes of nodes that are not traversable on a simple path from a start to
                an end.
        """
        self.not_traversable = []
        for idx in self.articulation_points:
            grid_copy = self.original_grid.copy()
            grid_copy[idx] = 0
            for idx_change in self.allowed_moves:
                new_idx = idx + idx_change
                if new_idx < 0 or new_idx >= self.n:
                    continue
                elif grid_copy[new_idx] != 1:
                    continue
                elif idx_change == 1 and new_idx % self.n_cols == 0:
                    continue
                elif idx_change == -1 and idx % self.n_cols == 0:
                    continue
                if new_idx == self.end or new_idx == self.start:
                    continue
                # There should be 'reset object' function or so, probably (?)
                bfs = BreadthFirstSearchFlat(grid_copy, self.n_cols, new_idx, self.end)
                end_reachable, _ = bfs.search_sssp()
                if not end_reachable:
                    bfs = BreadthFirstSearchFlat(grid_copy, self.n_cols, new_idx, self.start)
                    start_reachable, track_to_start = bfs.search_sssp()
                    if not start_reachable:
                        self.not_traversable = self.not_traversable + list(np.argwhere(bfs.grid == 2).flatten())

        self.not_traversable = list(set(self.not_traversable))

        return
