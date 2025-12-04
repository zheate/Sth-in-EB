from numpy import array, argmax, argmin, nan
from numpy import abs as abs_


def width_calculate_one_dimension(x_, y_, ratio):
    try:
        x = array(x_)
        y = array(y_)
        y_maximum_position = argmax(y)
        x_left = x[: y_maximum_position]
        x_right = x[y_maximum_position:]
        y_left = y[: y_maximum_position]
        y_right = y[y_maximum_position:]
        left_position = argmin(abs_(y_left - y.max() * ratio))
        right_position = argmin(abs_(y_right - y.max() * ratio))
        return x_right[right_position] - x_left[left_position]
    except ValueError:
        return nan
