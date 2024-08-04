import numpy
import torch
from radiomics import base, cMatrices

from torchradiomics.TorchRadiomicsBase import TorchRadiomicsBase


class TorchRadiomicsGLRLM(TorchRadiomicsBase):
  """
  RadiomicsGLRLM PyTorch implement
  """

  def __init__(self, inputImage, inputMask, **kwargs):
    super(TorchRadiomicsGLRLM, self).__init__(inputImage, inputMask, **kwargs)

    self.dtype = kwargs.get("dtype", torch.float64)
    self.device = kwargs.get("device", "cuda")

    self.weightingNorm = kwargs.get('weightingNorm', None)  # manhattan, euclidean, infinity

    self.P_glrlm = None
    self.imageArray = self._applyBinning(self.imageArray)

  def _initCalculation(self, voxelCoordinates=None):
    self.P_glrlm = self._calculateMatrix(voxelCoordinates)

    self._calculateCoefficients()

    self.logger.debug('GLRLM feature class initialized, calculated GLRLM with shape %s', self.P_glrlm.shape)

  def _calculateMatrix(self, voxelCoordinates=None):
    self.logger.debug('Calculating GLRLM matrix in C')

    Ng = self.coefficients['Ng']
    Nr = numpy.max(self.imageArray.shape)

    matrix_args = [
      self.imageArray,
      self.maskArray,
      Ng,
      Nr,
      self.settings.get('force2D', False),
      self.settings.get('force2Ddimension', 0)
    ]
    if self.voxelBased:
      matrix_args += [self.settings.get('kernelRadius', 1), voxelCoordinates]

    P_glrlm, angles = cMatrices.calculate_glrlm(*matrix_args)  # shape (Nvox, Ng, Nr, Na)
    P_glrlm = self.tensor(P_glrlm)
    angles = self.tensor(angles)

    self.logger.debug('Process calculated matrix')

    # Delete rows that specify gray levels not present in the ROI
    # NgVector = range(1, Ng + 1)  # All possible gray values
    GrayLevels = self.coefficients['grayLevels']  # Gray values present in ROI
    # emptyGrayLevels = numpy.array(list(set(NgVector) - set(GrayLevels)), dtype=int)  # Gray values NOT present in ROI

    # P_glrlm = numpy.delete(P_glrlm, emptyGrayLevels - 1, 1)
    P_glrlm = P_glrlm[:, GrayLevels - 1]

    # Optionally apply a weighting factor
    if self.weightingNorm is not None:
      self.logger.debug('Applying weighting (%s)', self.weightingNorm)

      pixelSpacing = self.inputImage.GetSpacing()[::-1]
      weights = torch.empty(len(angles))
      for a_idx, a in enumerate(angles):
        if self.weightingNorm == 'infinity':
          weights[a_idx] = max(torch.abs(a) * pixelSpacing)
        elif self.weightingNorm == 'euclidean':
          weights[a_idx] = torch.sqrt(torch.sum((torch.abs(a) * pixelSpacing) ** 2))
        elif self.weightingNorm == 'manhattan':
          weights[a_idx] = torch.sum(torch.abs(a) * pixelSpacing)
        elif self.weightingNorm == 'no_weighting':
          weights[a_idx] = 1
        else:
          self.logger.warning('weigthing norm "%s" is unknown, weighting factor is set to 1', self.weightingNorm)
          weights[a_idx] = 1

      P_glrlm = torch.sum(P_glrlm * weights[None, None, None, :], 3, keepdims=True)

    Nr = torch.sum(P_glrlm, (1, 2))

    # Delete empty angles if no weighting is applied
    if P_glrlm.shape[3] > 1:
      emptyAngles = torch.where(torch.sum(Nr, 0) == 0)
      if len(emptyAngles[0]) > 0:  # One or more angles are 'empty'
        # self.logger.debug('Deleting %d empty angles:\n%s', len(emptyAngles[0]), angles[emptyAngles])
        self.logger.debug('Deleting empty angles')
        P_glrlm = self.delete(P_glrlm, emptyAngles, 3)
        Nr = self.delete(Nr, emptyAngles, 1)
      else:
        self.logger.debug('No empty angles')

    Nr[Nr == 0] = torch.nan  # set sum to numpy.spacing(1) if sum is 0?
    self.coefficients['Nr'] = Nr

    return P_glrlm

  def _calculateCoefficients(self):
    self.logger.debug('Calculating GLRLM coefficients')

    pr = torch.sum(self.P_glrlm, 1)  # shape (Nvox, Nr, Na)
    pg = torch.sum(self.P_glrlm, 2)  # shape (Nvox, Ng, Na)

    ivector = self.tensor(self.coefficients['grayLevels'])  # shape (Ng,)
    jvector = torch.arange(1, self.P_glrlm.shape[2] + 1, dtype=self.dtype, device=self.device)  # shape (Nr,)

    # Delete columns that run lengths not present in the ROI
    emptyRunLenghts = torch.where(torch.sum(pr, (0, 2)) == 0)
    self.P_glrlm = self.delete(self.P_glrlm, emptyRunLenghts, 2)
    jvector = self.delete(jvector, emptyRunLenghts, 0)
    pr = self.delete(pr, emptyRunLenghts, 1)

    self.coefficients['pr'] = pr
    self.coefficients['pg'] = pg
    self.coefficients['ivector'] = ivector
    self.coefficients['jvector'] = jvector

  def getShortRunEmphasisFeatureValue(self):
    r"""
    **1. Short Run Emphasis (SRE)**

    .. math::
      \textit{SRE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\frac{\textbf{P}(i,j|\theta)}{j^2}}}{N_r(\theta)}

    SRE is a measure of the distribution of short run lengths, with a greater value indicative of shorter run lengths
    and more fine textural textures.
    """
    pr = self.coefficients['pr']
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']

    sre = torch.sum((pr / (jvector[None, :, None] ** 2)), 1) / Nr
    return torch.nanmean(sre, 1).cpu().numpy()

  def getLongRunEmphasisFeatureValue(self):
    r"""
    **2. Long Run Emphasis (LRE)**

    .. math::
      \textit{LRE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\textbf{P}(i,j|\theta)j^2}}{N_r(\theta)}

    LRE is a measure of the distribution of long run lengths, with a greater value indicative of longer run lengths and
    more coarse structural textures.
    """
    pr = self.coefficients['pr']
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']

    lre = torch.sum((pr * (jvector[None, :, None] ** 2)), 1) / Nr
    return torch.nanmean(lre, 1).cpu().numpy()

  def getGrayLevelNonUniformityFeatureValue(self):
    r"""
    **3. Gray Level Non-Uniformity (GLN)**

    .. math::
      \textit{GLN} = \frac{\sum^{N_g}_{i=1}\left(\sum^{N_r}_{j=1}{\textbf{P}(i,j|\theta)}\right)^2}{N_r(\theta)}

    GLN measures the similarity of gray-level intensity values in the image, where a lower GLN value correlates with a
    greater similarity in intensity values.
    """
    pg = self.coefficients['pg']
    Nr = self.coefficients['Nr']

    gln = torch.sum((pg ** 2), 1) / Nr
    return torch.nanmean(gln, 1).cpu().numpy()

  def getGrayLevelNonUniformityNormalizedFeatureValue(self):
    r"""
    **4. Gray Level Non-Uniformity Normalized (GLNN)**

    .. math::
      \textit{GLNN} = \frac{\sum^{N_g}_{i=1}\left(\sum^{N_r}_{j=1}{\textbf{P}(i,j|\theta)}\right)^2}{N_r(\theta)^2}

    GLNN measures the similarity of gray-level intensity values in the image, where a lower GLNN value correlates with a
    greater similarity in intensity values. This is the normalized version of the GLN formula.
    """
    pg = self.coefficients['pg']
    Nr = self.coefficients['Nr']

    glnn = torch.sum(pg ** 2, 1) / (Nr ** 2)
    return torch.nanmean(glnn, 1).cpu().numpy()

  def getRunLengthNonUniformityFeatureValue(self):
    r"""
    **5. Run Length Non-Uniformity (RLN)**

    .. math::
      \textit{RLN} = \frac{\sum^{N_r}_{j=1}\left(\sum^{N_g}_{i=1}{\textbf{P}(i,j|\theta)}\right)^2}{N_r(\theta)}

    RLN measures the similarity of run lengths throughout the image, with a lower value indicating more homogeneity
    among run lengths in the image.
    """
    pr = self.coefficients['pr']
    Nr = self.coefficients['Nr']

    rln = torch.sum((pr ** 2), 1) / Nr
    return torch.nanmean(rln, 1).cpu().numpy()

  def getRunLengthNonUniformityNormalizedFeatureValue(self):
    r"""
    **6. Run Length Non-Uniformity Normalized (RLNN)**

    .. math::
      \textit{RLNN} = \frac{\sum^{N_r}_{j=1}\left(\sum^{N_g}_{i=1}{\textbf{P}(i,j|\theta)}\right)^2}{N_r(\theta)^2}

    RLNN measures the similarity of run lengths throughout the image, with a lower value indicating more homogeneity
    among run lengths in the image. This is the normalized version of the RLN formula.
    """
    pr = self.coefficients['pr']
    Nr = self.coefficients['Nr']

    rlnn = torch.sum((pr ** 2), 1) / Nr ** 2
    return torch.nanmean(rlnn, 1).cpu().numpy()

  def getRunPercentageFeatureValue(self):
    r"""
    **7. Run Percentage (RP)**

    .. math::
      \textit{RP} = {\frac{N_r(\theta)}{N_p}}

    RP measures the coarseness of the texture by taking the ratio of number of runs and number of voxels in the ROI.

    Values are in range :math:`\frac{1}{N_p} \leq RP \leq 1`, with higher values indicating a larger portion of the ROI
    consists of short runs (indicates a more fine texture).

    .. note::
      Note that when weighting is applied and matrices are merged before calculation, :math:`N_p` is multiplied by
      :math:`n` number of matrices merged to ensure correct normalization (as each voxel is considered :math:`n` times)
    """
    pr = self.coefficients['pr']
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']

    Np = torch.sum(pr * jvector[None, :, None], 1)  # shape (Nvox, Na)

    rp = Nr / Np
    return torch.nanmean(rp, 1).cpu().numpy()

  def getGrayLevelVarianceFeatureValue(self):
    r"""
    **8. Gray Level Variance (GLV)**

    .. math::
      \textit{GLV} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_r}_{j=1}{p(i,j|\theta)(i - \mu)^2}

    Here, :math:`\mu = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_r}_{j=1}{p(i,j|\theta)i}`

    GLV measures the variance in gray level intensity for the runs.
    """
    ivector = self.coefficients['ivector']
    Nr = self.coefficients['Nr']
    pg = self.coefficients['pg'] / Nr[:, None, :]  # divide by Nr to get the normalized matrix

    u_i = torch.sum(pg * ivector[None, :, None], 1, keepdims=True)
    glv = torch.sum(pg * (ivector[None, :, None] - u_i) ** 2, 1)
    return torch.nanmean(glv, 1).cpu().numpy()

  def getRunVarianceFeatureValue(self):
    r"""
    **9. Run Variance (RV)**

    .. math::
      \textit{RV} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_r}_{j=1}{p(i,j|\theta)(j - \mu)^2}

    Here, :math:`\mu = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_r}_{j=1}{p(i,j|\theta)j}`

    RV is a measure of the variance in runs for the run lengths.
    """
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']
    pr = self.coefficients['pr'] / Nr[:, None, :]   # divide by Nr to get the normalized matrix

    u_j = torch.sum(pr * jvector[None, :, None], 1, keepdims=True)
    rv = torch.sum(pr * (jvector[None, :, None] - u_j) ** 2, 1)
    return torch.nanmean(rv, 1).cpu().numpy()

  def getRunEntropyFeatureValue(self):
    r"""
    **10. Run Entropy (RE)**

    .. math::
      \textit{RE} = -\displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_r}_{j=1}
      {p(i,j|\theta)\log_{2}(p(i,j|\theta)+\epsilon)}

    Here, :math:`\epsilon` is an arbitrarily small positive number (:math:`\approx 2.2\times10^{-16}`).

    RE measures the uncertainty/randomness in the distribution of run lengths and gray levels. A higher value indicates
    more heterogeneity in the texture patterns.
    """
    eps = torch.finfo(self.dtype).eps
    Nr = self.coefficients['Nr']
    p_glrlm = self.P_glrlm / Nr[:, None, None, :]  # divide by Nr to get the normalized matrix

    re = -torch.sum(p_glrlm * torch.log2(p_glrlm + eps), (1, 2))
    return torch.nanmean(re, 1).cpu().numpy()

  def getLowGrayLevelRunEmphasisFeatureValue(self):
    r"""
    **11. Low Gray Level Run Emphasis (LGLRE)**

    .. math::
      \textit{LGLRE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\frac{\textbf{P}(i,j|\theta)}{i^2}}}{N_r(\theta)}

    LGLRE measures the distribution of low gray-level values, with a higher value indicating a greater concentration of
    low gray-level values in the image.
    """
    pg = self.coefficients['pg']
    ivector = self.coefficients['ivector']
    Nr = self.coefficients['Nr']

    lglre = torch.sum((pg / (ivector[None, :, None] ** 2)), 1) / Nr
    return torch.nanmean(lglre, 1).cpu().numpy()

  def getHighGrayLevelRunEmphasisFeatureValue(self):
    r"""
    **12. High Gray Level Run Emphasis (HGLRE)**

    .. math::
      \textit{HGLRE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\textbf{P}(i,j|\theta)i^2}}{N_r(\theta)}

    HGLRE measures the distribution of the higher gray-level values, with a higher value indicating a greater
    concentration of high gray-level values in the image.
    """
    pg = self.coefficients['pg']
    ivector = self.coefficients['ivector']
    Nr = self.coefficients['Nr']

    hglre = torch.sum((pg * (ivector[None, :, None] ** 2)), 1) / Nr
    return torch.nanmean(hglre, 1).cpu().numpy()

  def getShortRunLowGrayLevelEmphasisFeatureValue(self):
    r"""
    **13. Short Run Low Gray Level Emphasis (SRLGLE)**

    .. math::
      \textit{SRLGLE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\frac{\textbf{P}(i,j|\theta)}{i^2j^2}}}{N_r(\theta)}

    SRLGLE measures the joint distribution of shorter run lengths with lower gray-level values.
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']

    srlgle = torch.sum((self.P_glrlm / ((ivector[None, :, None, None] ** 2) * (jvector[None, None, :, None] ** 2))),
                       (1, 2)) / Nr
    return torch.nanmean(srlgle, 1).cpu().numpy()

  def getShortRunHighGrayLevelEmphasisFeatureValue(self):
    r"""
    **14. Short Run High Gray Level Emphasis (SRHGLE)**

    .. math::
      \textit{SRHGLE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\frac{\textbf{P}(i,j|\theta)i^2}{j^2}}}{N_r(\theta)}

    SRHGLE measures the joint distribution of shorter run lengths with higher gray-level values.
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']

    srhgle = torch.sum((self.P_glrlm * (ivector[None, :, None, None] ** 2) / (jvector[None, None, :, None] ** 2)),
                       (1, 2)) / Nr
    return torch.nanmean(srhgle, 1).cpu().numpy()

  def getLongRunLowGrayLevelEmphasisFeatureValue(self):
    r"""
    **15. Long Run Low Gray Level Emphasis (LRLGLE)**

    .. math::
      \textit{LRLGLRE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\frac{\textbf{P}(i,j|\theta)j^2}{i^2}}}{N_r(\theta)}

    LRLGLRE measures the joint distribution of long run lengths with lower gray-level values.
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']

    lrlgle = torch.sum((self.P_glrlm * (jvector[None, None, :, None] ** 2) / (ivector[None, :, None, None] ** 2)),
                       (1, 2)) / Nr
    return torch.nanmean(lrlgle, 1).cpu().numpy()

  def getLongRunHighGrayLevelEmphasisFeatureValue(self):
    r"""
    **16. Long Run High Gray Level Emphasis (LRHGLE)**

    .. math::
      \textit{LRHGLRE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\textbf{P}(i,j|\theta)i^2j^2}}{N_r(\theta)}

    LRHGLRE measures the joint distribution of long run lengths with higher gray-level values.
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']

    lrhgle = torch.sum((self.P_glrlm * ((jvector[None, None, :, None] ** 2) * (ivector[None, :, None, None] ** 2))),
                       (1, 2)) / Nr
    return torch.nanmean(lrhgle, 1).cpu().numpy()
