def n_degree_curve(x, x_bounds, y_bounds, n, reverse=False):
    """
    Determine n-degree function on the segment with borders
    'x_bounds' and 'y_bounds' and get it's value depending on 'x'
    """

    assert len(x_bounds) == len(y_bounds) == 2, "'x' & 'y' bounds should be the same number"
    x_min, x_max = min(x_bounds), max(x_bounds)
    y_min, y_max = min(y_bounds), max(y_bounds)
    assert x_min <= x <= x_max, "'x' out of bounds"
    if reverse is True:
        return y_min + (y_max - y_min)*(1 - (abs(x) - x_min) / (x_max - x_min)) ** n
    return y_min + (y_max - y_min)*((abs(x) - x_min) / (x_max - x_min)) ** n
