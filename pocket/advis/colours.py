"""
Utilities related to colours

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
"""
import numpy as np
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap

__all__ = ["palette", "build_continuous_cmap", "build_preset_cmaps"]

def palette(n, dtype="float", **kwargs):
    c = sns.color_palette(n_colors=n, **kwargs)
    if dtype == "float":
        return c
    elif dtype == "uint8":
        return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in c]
    else:
        raise ValueError(f"Unsupported data type {dtype}.")

def build_continuous_cmap(rgb_x, rgb_v, zero_start=True, alpha_x=None, alpha_v=None, name=None):
    """
    Create a colour map by linearly interpolating between given colours.

    Parameters:
    -----------
    rgb_x: list[float]
        A list of values between 0 and 1 indicating interval boundaries.
    rgb_v: list[tuple]
        A list of RGB values between 0 and 1.
    zero_start: bool, default: True
        If True, insert (0, 0, 0) as the starting colour.
    alpha_x: list[float], default: None
        A list of values between 0 and 1 indicating interval boundaries.
        If left as None, [0, 1] will be used. 
    alpha_v: list[float], default: None
        A list of values between 0 and 1 indicating opacity level. If left
        as None, [0, 1] will be used.
    name: str, default: None
        Name of the created colour map. If left as None, "none" will be used.

    Returns:
    --------
    cmap: LinearSegmentedColormap
        Created colour map with linearly interpolated colours.

    Example:
    --------
    cmap = build_continuous_cmap(
        # Create two intervals: (0., 0.5), (0.5, 1.)
        [.5, 1.],
        # As zero_start is set to True, (0., 0., 0.) is the colour at level 0.
        [
            # The colour reached at level 0.5
            (1., .4980, .0549),
            # The colour reached at level 1.0
            (1., .7059, .5098),
        ],
        # Define one interval (0., 1.)
        alpha_x=[0., 1.,],
        # Opacity is increased from 0 to .8 linearly in the defined interval
        alpha_v=[0., .8,],
    )
    """
    rgb = np.asarray(rgb_v)
    r, g, b = np.split(rgb, 3, 1)
    if len(rgb_x) != len(r):
        raise ValueError("Defined interval boundaries rgb_x should have "
        "a length equal to the number of colours.")
    if zero_start:
        r = np.insert(r, 0, 0)
        g = np.insert(g, 0, 0)
        b = np.insert(b, 0, 0)
        rgb_x = np.insert(rgb_x, 0, 0)
    red = tuple((rgb_x[i], r[i], r[i]) for i in range(len(r)))
    green = tuple((rgb_x[i], g[i], g[i]) for i in range(len(g)))
    blue = tuple((rgb_x[i], b[i], b[i]) for i in range(len(b)))
    
    if alpha_x is None:
        alpha_x = [0., 1.]
    if alpha_v is None:
        alpha_v = [0., 1.]
    if len(alpha_x) != len(alpha_v):
        raise ValueError("Defined interval boundaries alpha_x should have "
        "a length equal to the number of alpha values.")
    alpha = tuple((alpha_x[i], alpha_v[i], alpha_v[i]) for i in range(len(alpha_x)))

    c_dict = {"red": red, "green": green, "blue": blue, "alpha":alpha}
    if name is None:
        name = "none"
    c_map = LinearSegmentedColormap(name, c_dict)

    return c_map

def build_preset_cmaps(n):
    default = palette(n)
    pastel = palette(n, palette="pastel")
    c_maps = [build_continuous_cmap(
        [.5, 1.], [default[i], pastel[i]], alpha_v=[0., .8]
        ) for i in range(n)]
    return c_maps