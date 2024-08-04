import radiomics
from radiomics import firstorder, glcm, gldm, glrlm, glszm, ngtdm

from .TorchRadiomicsFirstOrder import TorchRadiomicsFirstOrder
from .TorchRadiomicsGLCM import TorchRadiomicsGLCM
from .TorchRadiomicsGLDM import TorchRadiomicsGLDM
from .TorchRadiomicsGLRLM import TorchRadiomicsGLRLM
from .TorchRadiomicsGLSZM import TorchRadiomicsGLSZM
from .TorchRadiomicsNGTDM import TorchRadiomicsNGTDM


def inject_torch_radiomics():
    radiomics._featureClasses["firstorder"] = TorchRadiomicsFirstOrder
    radiomics._featureClasses["glcm"] = TorchRadiomicsGLCM
    radiomics._featureClasses["gldm"] = TorchRadiomicsGLDM
    radiomics._featureClasses["glrlm"] = TorchRadiomicsGLRLM
    radiomics._featureClasses["glszm"] = TorchRadiomicsGLSZM
    radiomics._featureClasses["ngtdm"] = TorchRadiomicsNGTDM


def restore_radiomics():
    radiomics._featureClasses["firstorder"] = firstorder.RadiomicsFirstOrder
    radiomics._featureClasses["glcm"] = glcm.RadiomicsGLCM
    radiomics._featureClasses["gldm"] = gldm.RadiomicsGLDM
    radiomics._featureClasses["glrlm"] = glrlm.RadiomicsGLRLM
    radiomics._featureClasses["glszm"] = glszm.RadiomicsGLSZM
    radiomics._featureClasses["ngtdm"] = ngtdm.RadiomicsNGTDM
