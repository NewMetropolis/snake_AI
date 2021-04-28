import numpy as np


def cell_traversable(grid, idx_arr):
    """Check if a cell/node is traversable i.e. has at least two edges."""
    directions = [np.array([0, 1], dtype=int), np.array([-1, 0], dtype=int), np.array([0, -1], dtype=int),
                  np.array([1, 0], dtype=int)]
    accessible = 0
    for direction in directions:
        if grid[tuple(idx_arr + direction)] != 0:
            accessible += 1

    return accessible > 1


def mark_start_end(grid, start, end):
    """"Mark start and end on a grid as 0's."""
    grid[start] = 0
    grid[end] = 0

    return


def fill_with_largest_rectangles(grid, only_traversable=True):
    """Fill grid with largest rectangles possible."""
    rectangles_dict = {}
    max_area = np.inf
    # Using three grids at the same time makes sens for debugging. Afterwards I should come with something more
    # efficient.
    filled_grid = grid.copy()
    zeroed_grid = grid.copy()
    # To avoid collision with other used mappings (0, 1) rectangle IDs start from 100.
    rectangle_id = 100
    while max_area > 1:
        max_area, max_row_stop, max_col_start, max_col_stop = find_largest_rectangle(zeroed_grid)
        if max_area == 1:
            rectangle_id += 1
            continue
        n_rows = max_area // (max_col_stop - max_col_start)
        max_row_start = max_row_stop - n_rows
        rectangles_dict[rectangle_id] = (max_area, max_row_start, max_row_stop, max_col_start, max_col_stop)
        zeroed_grid[max_row_start:max_row_stop, max_col_start:max_col_stop] = 0
        filled_grid[max_row_start:max_row_stop, max_col_start:max_col_stop] = rectangle_id
        rectangle_id += 1

    indexes_list = list(np.argwhere(zeroed_grid == 1))
    while indexes_list:
        idx_arr = indexes_list.pop()
        row_idx = idx_arr[0]
        col_idx = idx_arr[1]
        zeroed_grid[tuple(idx_arr)] = 0
        if only_traversable:
            if not cell_traversable(grid, idx_arr):
                continue
        rectangles_dict[rectangle_id] = (max_area, row_idx, row_idx, col_idx, col_idx)
        filled_grid[tuple(idx_arr)] = rectangle_id
        rectangle_id += 1

    return


def find_largest_rectangle(grid):
    """Find the largest rectangle on a grid."""
    max_area = 0
    max_row_stop = -1
    max_col_start = -1
    max_col_stop = -1
    cum_sum_corrected = zero_flushed_cumsum(grid)

    for row_id, current_row in enumerate(cum_sum_corrected):
        area, area_col_start, area_col_stop = largest_area_under_histogram(current_row)
        if area > max_area:
            max_area = area
            max_row_stop = row_id + 1
            max_col_start = area_col_start
            max_col_stop = area_col_stop

    return max_area, max_row_stop, max_col_start, max_col_stop


def zero_flushed_cumsum(grid):
    """Output column-wise cumulative sum that resets to zero every time a 0 occurs in a cell."""
    zeros_location = grid == 0
    cum_sum = np.cumsum(grid, axis=0)
    grid_with_correction = grid.copy()
    grid_with_correction[zeros_location] = np.negative(cum_sum[zeros_location])
    cum_sum_corrected = np.cumsum(grid_with_correction, axis=0)

    return cum_sum_corrected


def largest_area_under_histogram(histogram):
    """Output the largest area under a given histogram/1D array together with starting and ending bar/column numbers."""
    stack = list()
    index = 0
    max_area = 0
    while index < len(histogram):
        if (not stack) or (histogram[stack[-1]] <= histogram[index]):
            stack.append(index)
            index += 1
        else:
            top_of_stack = stack.pop()
            area = (histogram[top_of_stack] * ((index - stack[-1] - 1) if stack else index))
            max_area = max(max_area, area)
            if area == max_area:
                if stack:
                    max_col_start = top_of_stack
                    max_col_stop = max_col_start + index - stack[-1] - 1
                else:
                    max_col_start = 0
                    max_col_stop = index
                    # max_col_start = top_of_stack
                    # max_col_stop = max_col_start + index
    while stack:
        top_of_stack = stack.pop()
        area = (histogram[top_of_stack] * ((index - stack[-1] - 1) if stack else index))
        max_area = max(max_area, area)
        if area == max_area:
            if stack:
                max_col_start = top_of_stack
                max_col_stop = max_col_start + index - stack[-1] - 1
            else:
                max_col_start = 0
                max_col_stop = index
                # max_col_start = top_of_stack
                # max_col_stop = max_col_start + index

    return max_area, max_col_start, max_col_stop
