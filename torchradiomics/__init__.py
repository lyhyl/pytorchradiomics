import radiomics
from radiomics import firstorder, glcm, glrlm, ngtdm
from .TorchRadiomicsFirstOrder import TorchRadiomicsFirstOrder
from .TorchRadiomicsGLCM import TorchRadiomicsGLCM
from .TorchRadiomicsGLRLM import TorchRadiomicsGLRLM
from .TorchRadiomicsNGTDM import TorchRadiomicsNGTDM


def inject_torch_radiomics():
    radiomics._featureClasses["firstorder"] = TorchRadiomicsFirstOrder
    radiomics._featureClasses["glcm"] = TorchRadiomicsGLCM
    radiomics._featureClasses["glrlm"] = TorchRadiomicsGLRLM
    radiomics._featureClasses["ngtdm"] = TorchRadiomicsNGTDM


def restore_radiomics():
    radiomics._featureClasses["firstorder"] = firstorder.RadiomicsFirstOrder
    radiomics._featureClasses["glcm"] = glcm.RadiomicsGLCM
    radiomics._featureClasses["glrlm"] = glrlm.RadiomicsGLRLM
    radiomics._featureClasses["ngtdm"] = ngtdm.RadiomicsNGTDM
