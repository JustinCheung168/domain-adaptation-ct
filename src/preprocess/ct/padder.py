from typing import Tuple

# Image manipulation
import numpy as np

class Padder:
  """2D image padder"""
  def __init__(self, pad_sz = Tuple[Tuple[int, int], Tuple[int, int]]):
    self.pad_sz = pad_sz

  def pad(self, arr: np.ndarray):
    return np.pad(arr, self.pad_sz, mode="constant", constant_values=0)

  def unpad(self, arr: np.ndarray):
    return arr[self.pad_sz:-self.pad_sz, self.pad_sz:-self.pad_sz]
  
  @staticmethod
  def calculate_diagonal_length(shape: Tuple[int, int]) -> int:
    """
    Calculate the length of the diagonal for an image of dimensions `shape`.
    (same format as a np.ndarray's `.shape`.)
    """
    max_dimension = max(shape)
    diagonal = np.sqrt(2) * max_dimension
    return int(np.ceil((diagonal - max_dimension) / 2))

class SymmetricPadder(Padder):
  """2D image padder"""
  def __init__(self, pad_sz = int):
    super().__init__(((pad_sz, pad_sz), (pad)))
