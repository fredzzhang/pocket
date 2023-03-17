"""
Utilities related to text overlay

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
"""

import matplotlib.pyplot as plt
import matplotlib.patheffects as peff

def text(x, y, t, plot=None, text_args=None, effect_args=None):
    """
    Plot text onto a figure.

    x: float
        x coordinate for the text position.
    y: float
        y coordinate for the text position.
    t: str
        Text content.
    plot: tuple(Figure, AxesSubplot), default: None
        The figure and axis of a given plot. If left as None, a new figure will be created.
    text_args: dict, default: None
        Keyworded arguments for plt.text(). If left as None, `fontsize=15`, `fontweight="semibold"`
        and `color="w"` will be used.
    effect_args: dict, default: None
        Keyworded arguments for set_path_effects(). If left as None, `linewidth=5` and `foreground="k"`
        will be used.

    Returns:
    --------
    ax: AxesSubplot
        Axis the text was plotted with.
    """
    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot

    if text_args is None:
        text_args = {"fontsize": 15, "fontweight": "semibold", "color": "w"}
    if effect_args is None:
        effect_args = {"linewidth": 5, "foreground": "k"}
    txt = ax.text(x, y, t, **text_args)
    txt.set_path_effects([peff.withStroke(**effect_args)])
    fig.canvas.draw()
    renderer = fig.canvas.renderer
    ax.draw(renderer)

    return ax