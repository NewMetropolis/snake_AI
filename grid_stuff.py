import numpy as np


def fill_with_largest_rectangles(grid):
    """Fill grid with largest rectangles possible."""
    max_area = np.inf
    while max_area > 1:
        max_area, max_row_stop, max_col_start, max_col_stop = find_largest_rectangle(grid)
        n_rows = max_area // (max_col_stop - max_col_start + 1)
        max_row_start = max_row_stop - n_rows + 1
        grid[max_row_start:max_row_stop + 1, max_col_start:max_col_stop+1] = 0

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
            max_row_stop = row_id
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
                    max_col_start = stack[-1] + 1
                    max_col_stop = index - 1
                else:
                    max_col_start = 0
                    max_col_stop = index - 1
    while stack:
        top_of_stack = stack.pop()
        area = (histogram[top_of_stack] * ((index - stack[-1] - 1) if stack else index))
        max_area = max(max_area, area)
        if stack:
            max_col_start = stack[-1] + 1
            max_col_stop = index - 1
        else:
            max_col_start = 0
            max_col_stop = index - 1

    return max_area, max_col_start, max_col_stop
