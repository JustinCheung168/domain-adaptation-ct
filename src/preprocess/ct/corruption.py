from typing import List

# Image manipulation
import numpy as np

class SinogramCorruptor:
  """
  Adds an artifact in the sinogram.
  """
  def create_multiplicative_ring_artifact(self, sinogram: np.ndarray, detectors: List[int], factors: List[float]):
    """
    Create a ring artifact characterized by a detector only receiving a certain
    fraction of incoming X-rays.

    For example, setting `detectors` to [5, 7] and `factors` to [0.2, 0.5]
    causes detector at index 5 to receive only 20% of the incoming X-rays,
    and detector at index 7 to receive only 50% of the incoming X-rays.

    Args:
      sinogram: A 2D array of shape (n_angles, n_detectors) representing the
        sinogram.
      detectors: A list of detector indices that will receive the artifact.
      factors: A list of multiplicative factors by which to multiply each
        detector. One element in `factors` corresponds to one index in
        `detectors`.

    Returns:
      A 2D array of shape (n_angles, n_detectors) representing the corrupted sinogram.
    """
    assert len(detectors) == len(factors), "Must provide a factor for each detector"

    corrupted_sinogram = sinogram.copy()
    for i in range(len(detectors)):
      corrupted_sinogram[:, detectors[i]] = corrupted_sinogram[:, detectors[i]] * factors[i]

    return corrupted_sinogram