# PyTorchRadiomics

PyTorch implementation of [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) Extractor

# Performance Improvement

It can speed up voxel-based features extraction significantly, especially GLCM features.

Using it to extract non-voxel-based features is *NOT* recommended (it is slower).

## Voxel-based Features Extraction Performance Comparison
Intel i9-10900K v.s. RTX 3080 10G (dtype=torch.float64), Size=$16^3$

|Type|CPU Time|Torch Time|Max Abs. Error|Max Rel. Error|
|-|-|-|-|-|
GLCM|636s|23.8s|2.32e-09|7.92e-12|
FirstOrder|4.3s|0.244s|2.84e-14|2.22e-16|
GLRLM|1.71s|0.731s|2.72e-12|8.88e-16|
NGTDM|4.03s|0.398s|3.27e-11|3.99e-15|

# Installation
```
pip install pytorchradiomics
```

# Usage

Only two extra keyword arguments:
1. `device`: `str` or `torch.device`, default: `"cuda"`
2. `dtype`: `torch.dtype`, default: `torch.float64`

Direct usage:
```python
from torchradiomics import (TorchRadiomicsFirstOrder, TorchRadiomicsGLCM,
                            TorchRadiomicsGLRLM, TorchRadiomicsNGTDM,
                            inject_torch_radiomics, restore_radiomics)

ext = TorchRadiomicsGLCM(
    img_norm, mask_norm,
    voxelBased=True, padDistance=kernel,
    kernelRadius=kernel, maskedKernel=False, voxelBatch=512,
    dtype=torch.float64, # it is default
    device="cuda:0",
    **get_default_settings())

features = ext.execute()
```

Or use injection to use `RadiomicsFeatureExtractor`:

```python
from radiomics.featureextractor import RadiomicsFeatureExtractor
from torchradiomics import (TorchRadiomicsFirstOrder, TorchRadiomicsGLCM,
                            TorchRadiomicsGLRLM, TorchRadiomicsNGTDM,
                            inject_torch_radiomics, restore_radiomics)

inject_torch_radiomics() # replace cpu version with torch version

ext = RadiomicsFeatureExtractor(
    voxelBased=True, padDistance=kernel,
    kernelRadius=kernel, maskedKernel=False, voxelBatch=512,
    dtype=torch.float64, # it is default
    device="cuda:0",
    **get_default_settings())
ext.execute(img, mask, voxelBased=True)

restore_radiomics() # restore
```
