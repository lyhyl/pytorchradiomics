import logging
import warnings
from typing import Type
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import radiomics
import radiomics.base
import SimpleITK as sitk
import torch
from pyinstrument import Profiler
from radiomics import firstorder, glcm, glrlm, ngtdm
from radiomics.featureextractor import RadiomicsFeatureExtractor
from radiomics.imageoperations import checkMask, cropToTumorMask
from tqdm import tqdm

from torchradiomics import (TorchRadiomicsFirstOrder, TorchRadiomicsGLCM,
                            TorchRadiomicsGLRLM, TorchRadiomicsNGTDM,
                            inject_torch_radiomics, restore_radiomics)


def get_default_settings():
    settings = {}
    settings["normalize"] = True
    settings["normalizeScale"] = 100
    settings["binWidth"] = 5
    settings["voxelArrayShift"] = 300
    return settings


def diff(a: dict, b: dict):
    assert len(a.keys()) == len(b.keys())
    pairs = [(sitk.GetArrayFromImage(a[n]), sitk.GetArrayFromImage(b[n])) for n in a.keys()]
    print("all close:", all(np.allclose(x, y) for x, y in pairs))
    print("max abs diff:", max(np.max(np.abs((x - y))) for x, y in pairs))
    print("max rel diff1:", max(np.nanmax(np.abs((1 - x / y))) for x, y in pairs))
    print("max rel diff2:", max(np.nanmax(np.abs((1 - y / x))) for x, y in pairs))


def test(img, mask, kernel,
         typeA: Type[radiomics.base.RadiomicsFeaturesBase],
         typeB: Type[radiomics.base.RadiomicsFeaturesBase]):
    rf_ext = RadiomicsFeatureExtractor(
        voxelBased=True, padDistance=kernel,
        kernelRadius=kernel, maskedKernel=False, voxelBatch=512,
        **get_default_settings())
    img_norm, mask_norm = rf_ext.loadImage(img, mask, None, **rf_ext.settings)

    a_ext = typeA(
        img_norm, mask_norm,
        voxelBased=True, padDistance=kernel,
        kernelRadius=kernel, maskedKernel=False, voxelBatch=1024,
        **get_default_settings())

    b_ext = typeB(
        img_norm, mask_norm,
        voxelBased=True, padDistance=kernel,
        kernelRadius=kernel, maskedKernel=False, voxelBatch=512,
        dtype=torch.float64,
        **get_default_settings())
        
    profiler1 = Profiler()
    profiler1.start()
    a = a_ext.execute()
    profiler1.stop()
    profiler1.open_in_browser()

    profiler2 = Profiler()
    profiler2.start()
    b = b_ext.execute()
    profiler2.stop()
    profiler2.open_in_browser()

    diff(a, b)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*OMP_NUM_THREADS=\d.*") 
    warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*") 

    radiomics.setVerbosity(logging.INFO)  # Verbosity must be at least INFO to enable progress bar
    radiomics.progressReporter = tqdm
    
    kernel = 1
    size = 16
    img = sitk.GetImageFromArray(np.random.randn(size, size, size))
    mask = sitk.GetImageFromArray(np.ones((size, size, size)))

    test(img, mask, kernel, glcm.RadiomicsGLCM, TorchRadiomicsGLCM)
    test(img, mask, kernel, firstorder.RadiomicsFirstOrder, TorchRadiomicsFirstOrder)
    test(img, mask, kernel, glrlm.RadiomicsGLRLM, TorchRadiomicsGLRLM)
    test(img, mask, kernel, ngtdm.RadiomicsNGTDM, TorchRadiomicsNGTDM)

    # inject_torch_radiomics()
        
    # ext = RadiomicsFeatureExtractor(
    #     voxelBased=True, padDistance=kernel,
    #     kernelRadius=kernel, maskedKernel=False, voxelBatch=512,
    #     **get_default_settings())
    # profiler = Profiler()
    # profiler.start()
    # ext.execute(img, mask, voxelBased=True)
    # profiler.stop()
    # profiler.open_in_browser()

    # restore_radiomics()
    