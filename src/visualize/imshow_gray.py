"""Custom matplotlib wrapper for this application."""

from typing import Optional, Tuple, List
from matplotlib.image import AxesImage

# Image display
import matplotlib.pyplot as plt

# Image manipulation
import numpy as np

class ImShowGray:
  @staticmethod
  def imshow(img: np.ndarray, 
            axes: Optional[plt.Axes] = None,
            title: Optional[str] = None,
            window: Optional[Tuple[float, float]] = None,
            title_stats: bool = False) -> plt.Axes:
    """
    Display grayscale image with preferred settings.

    Window is specified as (vmin, vmax) - so the minimum and maximum intensity values.
    Values under vmin will be visualized as black,
    whereas values above vmax will be visualized as white.
    Recommend window=(0, 255) for 8-bit images.
    """
    if (axes is None):
      _, axes = plt.subplots(1, 1)
    
    if title is None:
      if title_stats:
        axes.set_title(ImShowGray.stat_string(img))
      else:
        axes.set_title(title)
    else:
      if title_stats:
        axes.set_title(title+"\n"+ImShowGray.stat_string(img))
      else:
        axes.set_title(title)

    if window is None:
      window = (None, None)

    axes.imshow(img, cmap='gray', vmin=window[0], vmax=window[1])
    axes.axis('off')

    return axes

  @staticmethod
  def stat_string(img: np.ndarray) -> str:
    """Produce a string with common simple statistics of interest"""
    return f"Max: {np.nanmax(img):.4f}" + \
           f"\nMin: {np.nanmin(img):.4f}" + \
           f"\nMean: {np.nanmean(img):.4f}" + \
           f"\nStdev: {np.nanstd(img):.4f}" + \
           f"\n#NaN: {np.isnan(img).sum()}"

  @staticmethod
  def imshow_diff(img_new: np.ndarray, img_old: np.ndarray, 
                  window: Optional[Tuple[float, float]] = None,
                  diff_window: Optional[Tuple[float, float]] = None,
                  titles: Optional[Tuple[str, str]] = None,
                  title_stats: bool = False) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Display the difference between two images for comparative purposes.
    The difference taken is img_new minus img_old.
    """
    if titles is None:
      titles = ("New", "Old")

    img_diff = img_new - img_old
    img_absdiff = np.abs(img_diff)

    fig, axes = plt.subplots(1, 4, figsize=(12, 6))

    ImShowGray.imshow(img_new, axes[0], title=titles[0], title_stats=title_stats, window=window)
    ImShowGray.imshow(img_old, axes[1], title=titles[1], title_stats=title_stats, window=window)
    ImShowGray.imshow(img_diff, axes[2], title=f"{titles[0]} - {titles[1]}", title_stats=title_stats, window=diff_window)
    ImShowGray.imshow(img_absdiff, axes[3], title=f"|{titles[0]} - {titles[1]}|", title_stats=title_stats, window=diff_window)

    return fig, axes
