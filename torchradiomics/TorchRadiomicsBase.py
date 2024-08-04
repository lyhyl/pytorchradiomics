import numpy
import torch
from radiomics import base


class TorchRadiomicsBase(base.RadiomicsFeaturesBase):
  """
  RadiomicsGLCM PyTorch implement
  """
  def __init__(self, inputImage, inputMask, **kwargs):
    super(TorchRadiomicsBase, self).__init__(inputImage, inputMask, **kwargs)

    self.dtype = kwargs.get("dtype", torch.float64)
    self.device = kwargs.get("device", "cuda")
  
  def tensor(self, array: numpy.ndarray):
    return torch.tensor(array, dtype=self.dtype, device=self.device)
  
  def delete(self, arr: torch.Tensor, ind: int | tuple | list, dim: int) -> torch.Tensor:
    """
    https://gist.github.com/velikodniy/6efef837e67aee2e7152eb5900eb0258
    """
    # skip = [i for i in range(arr.size(dim)) if i != ind]
    if isinstance(ind, int):
      skip = [i for i in range(arr.size(dim)) if i != ind]
    elif isinstance(ind, (tuple, list)): # torch.where
      lst: torch.Tensor = ind[0]
      lst = lst.cpu().numpy()
      skip = [i for i in range(arr.size(dim)) if i not in lst]
    else:
      raise TypeError("ind wrong type")
    indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
    return arr.__getitem__(indices)