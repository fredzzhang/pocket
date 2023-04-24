"""
Utilities related to heatmaps

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
"""

import torch
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

from .colours import build_preset_cmaps

def heatmap(image, heatmaps, ax=None, alpha=.6, c_maps=None, interp_args=None, save_path=None):
    """
    Overlay heatmaps on images and save the result.

    Parameters:
    -----------
    image: PIL.Image
        Source image to overlay the heatmaps on.
    heatmaps: Tensor
        Heatmap tensors of shape (N, H, W). For N>1, by default, different colour maps will
        be used for each heatmap.
    ax: AxesSubplot, default: None
        Axis to plot the image and heatmaps with. If left as None, a new figure will be created.
    alpha: float, default: .6
        Opacity level for the first heatmap. If set to None, the heatmap will be directly overlaid
        onto the image without blending.
    c_maps: LinearSegmentedColormap, default: None
        Colour maps used to visualise the heatmaps. If left as None, default colour maps from
        pocket.advis.build_preset_cmaps will be used. If otherwise specified, it should be 
        either a list with the length equal to the number of heatmaps, or a single colour map.
    interp_args: dict, default: None
        Arguments for heatmap interpolation. If left as None, `bicubic` and `align_corners=True`
        will be used.
    save_path: str, default: None
        Path for the saved figure. Note that image file extension should also be included.
    
    Returns:
    --------
    ax: AxesSubplot
        Axis the image and heatmaps were plotted with.
    """
    w, h = image.size
    if heatmaps.ndim != 3:
        raise ValueError("The heatmap tensor should have three dimensions (N, H, W).")

    if c_maps is None:
        c_maps = build_preset_cmaps(heatmaps.shape[0])
    elif type(c_maps) is not list:
        c_maps = [c_maps for _ in range(heatmaps.shape[0])]
    if interp_args is None:
        interp_args = {"mode": "bicubic", "align_corners": True}
    if heatmaps.shape[1:] != torch.Size([h, w]):
        heatmaps = F.interpolate(heatmaps.unsqueeze(0), size=(h, w), **interp_args).squeeze(0)
        heatmaps.clamp_(min=0, max=1)

    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(image)
    for i in range(heatmaps.shape[0]):
        hm = heatmaps[i]
        # Set opacity level for the first heatmap only. The rest
        # will be directly overlaied on top of the previous ones.
        opacity = alpha if i == 0 else None
        sns.heatmap(
            hm.numpy(), alpha=opacity, ax=ax, cmap=c_maps[i],
            xticklabels=False, yticklabels=False, cbar=False
        )
    # Remove the figure margin.
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    return ax