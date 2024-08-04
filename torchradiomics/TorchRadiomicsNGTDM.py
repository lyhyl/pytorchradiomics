import numpy
import torch
from radiomics import base, cMatrices

from torchradiomics.TorchRadiomicsBase import TorchRadiomicsBase


class TorchRadiomicsNGTDM(TorchRadiomicsBase):
  """
  RadiomicsNGTDM PyTorch implement
  """

  def __init__(self, inputImage, inputMask, **kwargs):
    super(TorchRadiomicsNGTDM, self).__init__(inputImage, inputMask, **kwargs)

    self.dtype = kwargs.get("dtype", torch.float64)
    self.device = kwargs.get("device", "cuda")

    self.P_ngtdm = None
    self.imageArray = self._applyBinning(self.imageArray)

  def _initCalculation(self, voxelCoordinates=None):
    self.P_ngtdm = self._calculateMatrix(voxelCoordinates)
    self._calculateCoefficients()

  def _calculateMatrix(self, voxelCoordinates=None):
    matrix_args = [
      self.imageArray,
      self.maskArray,
      numpy.array(self.settings.get('distances', [1])),
      self.coefficients['Ng'],
      self.settings.get('force2D', False),
      self.settings.get('force2Ddimension', 0)
    ]
    if self.voxelBased:
      matrix_args += [self.settings.get('kernelRadius', 1), voxelCoordinates]

    P_ngtdm = cMatrices.calculate_ngtdm(*matrix_args)  # shape (Nvox, Ng, 3)
    P_ngtdm = self.tensor(P_ngtdm)

    # Delete empty grey levels
    emptyGrayLevels = torch.where(torch.sum(P_ngtdm[:, :, 0], 0) == 0)
    P_ngtdm = self.delete(P_ngtdm, emptyGrayLevels, 1)

    return P_ngtdm

  def _calculateCoefficients(self):
    # No of voxels that have a valid region, lesser equal to Np
    Nvp = torch.sum(self.P_ngtdm[:, :, 0], 1)  # shape (Nvox,)
    self.coefficients['Nvp'] = Nvp  # shape (Nv,)

    # Normalize P_ngtdm[:, 0] (= n_i) to obtain p_i
    self.coefficients['p_i'] = self.P_ngtdm[:, :, 0] / Nvp[:, None]

    self.coefficients['s_i'] = self.P_ngtdm[:, :, 1]
    self.coefficients['ivector'] = self.P_ngtdm[:, :, 2]

    # Ngp = number of graylevels, for which p_i > 0
    self.coefficients['Ngp'] = torch.sum(self.P_ngtdm[:, :, 0] > 0, 1)

    p_zero = torch.where(self.coefficients['p_i'] == 0)
    self.coefficients['p_zero'] = p_zero

  def getCoarsenessFeatureValue(self):
    r"""
    Calculate and return the coarseness.

    :math:`Coarseness = \frac{1}{\sum^{N_g}_{i=1}{p_{i}s_{i}}}`

    Coarseness is a measure of average difference between the center voxel and its neighbourhood and is an indication
    of the spatial rate of change. A higher value indicates a lower spatial change rate and a locally more uniform texture.

    N.B. :math:`\sum^{N_g}_{i=1}{p_{i}s_{i}}` potentially evaluates to 0 (in case of a completely homogeneous image).
    If this is the case, an arbitrary value of :math:`10^6` is returned.
    """
    p_i = self.coefficients['p_i']
    s_i = self.coefficients['s_i']
    sum_coarse = torch.sum(p_i * s_i, 1)

    sum_coarse[sum_coarse != 0] = 1 / sum_coarse[sum_coarse != 0]
    sum_coarse[sum_coarse == 0] = 1e6
    return sum_coarse.cpu().numpy()

  def getContrastFeatureValue(self):
    r"""
    Calculate and return the contrast.

    :math:`Contrast = \left(\frac{1}{N_{g,p}(N_{g,p}-1)}\displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{p_{i}p_{j}(i-j)^2}\right)
    \left(\frac{1}{N_{v,p}}\displaystyle\sum^{N_g}_{i=1}{s_i}\right)\text{, where }p_i \neq 0, p_j \neq 0`

    Contrast is a measure of the spatial intensity change, but is also dependent on the overall gray level dynamic range.
    Contrast is high when both the dynamic range and the spatial change rate are high, i.e. an image with a large range
    of gray levels, with large changes between voxels and their neighbourhood.

    N.B. In case of a completely homogeneous image, :math:`N_{g,p} = 1`, which would result in a division by 0. In this
    case, an arbitray value of 0 is returned.
    """
    Ngp = self.coefficients['Ngp']  # shape (Nvox,)
    Nvp = self.coefficients['Nvp']  # shape (Nvox,)
    p_i = self.coefficients['p_i']  # shape (Nvox, Ng)
    s_i = self.coefficients['s_i']  # shape (Nvox, Ng)
    i = self.coefficients['ivector']  # shape (Ng,)

    div = Ngp * (Ngp - 1)

    # Terms where p_i = 0 or p_j = 0 will calculate as 0, and therefore do not affect the sum
    contrast = (torch.sum(p_i[:, :, None] * p_i[:, None, :] * (i[:, :, None] - i[:, None, :]) ** 2, (1, 2)) *
                torch.sum(s_i, 1) / Nvp)

    contrast[div != 0] /= div[div != 0]
    contrast[div == 0] = 0

    return contrast.cpu().numpy()

  def getBusynessFeatureValue(self):
    r"""
    Calculate and return the busyness.

    :math:`Busyness = \frac{\sum^{N_g}_{i = 1}{p_{i}s_{i}}}{\sum^{N_g}_{i = 1}\sum^{N_g}_{j = 1}{|ip_i - jp_j|}}\text{, where }p_i \neq 0, p_j \neq 0`

    A measure of the change from a pixel to its neighbour. A high value for busyness indicates a 'busy' image, with rapid
    changes of intensity between pixels and its neighbourhood.

    N.B. if :math:`N_{g,p} = 1`, then :math:`busyness = \frac{0}{0}`. If this is the case, 0 is returned, as it concerns
    a fully homogeneous region.
    """
    p_i = self.coefficients['p_i']  # shape (Nv, Ngp)
    s_i = self.coefficients['s_i']  # shape (Nv, Ngp)
    i = self.coefficients['ivector']  # shape (Nv, Ngp)
    p_zero = self.coefficients['p_zero']  # shape (2, z)

    i_pi = i * p_i
    absdiff = torch.abs(i_pi[:, :, None] - i_pi[:, None, :])

    # Remove terms from the sum where p_i = 0 or p_j = 0
    absdiff[p_zero[0], :, p_zero[1]] = 0
    absdiff[p_zero[0], p_zero[1], :] = 0

    absdiff = torch.sum(absdiff, (1, 2))

    busyness = torch.sum(p_i * s_i, 1)
    busyness[absdiff != 0] = busyness[absdiff != 0] / absdiff[absdiff != 0]
    busyness[absdiff == 0] = 0
    return busyness.cpu().numpy()

  def getComplexityFeatureValue(self):
    r"""
    Calculate and return the complexity.

    :math:`Complexity = \frac{1}{N_{v,p}}\displaystyle\sum^{N_g}_{i = 1}\displaystyle\sum^{N_g}_{j = 1}{|i - j|
    \frac{p_{i}s_{i} + p_{j}s_{j}}{p_i + p_j}}\text{, where }p_i \neq 0, p_j \neq 0`

    An image is considered complex when there are many primitive components in the image, i.e. the image is non-uniform
    and there are many rapid changes in gray level intensity.
    """
    Nvp = self.coefficients['Nvp']  # shape (Nv,)
    p_i = self.coefficients['p_i']  # shape (Nv, Ngp)
    s_i = self.coefficients['s_i']  # shape (Nv, Ngp)
    i = self.coefficients['ivector']  # shape (Nv, Ngp)
    p_zero = self.coefficients['p_zero']  # shape (2, z)

    pi_si = p_i * s_i
    numerator = pi_si[:, :, None] + pi_si[:, None, :]

    # Remove terms from the sum where p_i = 0 or p_j = 0
    numerator[p_zero[0], :, p_zero[1]] = 0
    numerator[p_zero[0], p_zero[1], :] = 0

    divisor = p_i[:, :, None] + p_i[:, None, :]
    divisor[divisor == 0] = 1  # Prevent division by 0 errors. (Numerator is 0 at those indices too)

    complexity = torch.sum(torch.abs(i[:, :, None] - i[:, None, :]) * numerator / divisor, (1, 2)) / Nvp

    return complexity.cpu().numpy()

  def getStrengthFeatureValue(self):
    r"""
    Calculate and return the strength.

    :math:`Strength = \frac{\sum^{N_g}_{i = 1}\sum^{N_g}_{j = 1}{(p_i + p_j)(i-j)^2}}{\sum^{N_g}_{i = 1}{s_i}}\text{, where }p_i \neq 0, p_j \neq 0`

    Strength is a measure of the primitives in an image. Its value is high when the primitives are easily defined and
    visible, i.e. an image with slow change in intensity but more large coarse differences in gray level intensities.

    N.B. :math:`\sum^{N_g}_{i=1}{s_i}` potentially evaluates to 0 (in case of a completely homogeneous image).
    If this is the case, 0 is returned.
    """
    p_i = self.coefficients['p_i']  # shape (Nv, Ngp)
    s_i = self.coefficients['s_i']  # shape (Nv, Ngp)
    i = self.coefficients['ivector']  # shape (Nv, Ngp)
    p_zero = self.coefficients['p_zero']  # shape (2, z)

    sum_s_i = torch.sum(s_i, 1)

    strength = (p_i[:, :, None] + p_i[:, None, :]) * (i[:, :, None] - i[:, None, :]) ** 2

    # Remove terms from the sum where p_i = 0 or p_j = 0
    strength[p_zero[0], :, p_zero[1]] = 0
    strength[p_zero[0], p_zero[1], :] = 0

    strength = torch.sum(strength, (1, 2))
    strength[sum_s_i != 0] /= sum_s_i[sum_s_i != 0]
    strength[sum_s_i == 0] = 0

    return strength.cpu().numpy()
