import pyCT
from pyCT.parameter import _Parameters
from .projectionCPU import *
from projectionGPU import projectParallelBeamGPU, projectConeBeamGPU
from copy import deepcopy

def project(object_array : np.ndarray,
            parameters   : _Parameters,
            **kwargs):
    '''
    key : cuda, ray_step
    '''
    # check CUDA
    is_cuda = False if pyCT.CUDA is None else True        
    if 'cuda' in kwargs.keys():
        if is_cuda:
            if not kwargs['cuda']:
                is_cuda = False
        else:
            if kwargs['cuda']:
                print('CUDA is not available ...')
    
    # set step size
    if 'ray_step' in kwargs.keys():
        ray_step = float(kwargs['ray_step'])
    else:
        ray_step = .5
    
    # get parameters
    mode = parameters.mode
    s2d = parameters.source.distance.source2detector
    nx, ny, nz = parameters.object.size.get()
    su, sv = parameters.detector.length.get()
    nu, nv = parameters.detector.size.get()
    na = len(parameters.source.motion.rotation.get()[0])
    near, far, nw = _getNearFar(parameters, ray_step)
    
    # get transformation
    transformation = pyCT.getTransformation(parameters, nw, near, far)
    transformationMatrix = transformation.getForward()

    # run
    if is_cuda:
        detector_array = np.zeros(na*nv*nu, dtype=np.float32)
        object_array = object_array.flatten().astype(np.float32)
        transformationMatrix = transformationMatrix.flatten().astype(np.float32)
        if mode:
            ou, ov = parameters.detector.motion.translation.get(axis=0).astype(np.float32)
            oa = parameters.detector.motion.rotation.get().astype(np.float32)
            if (len(ou) == 1) and (len(ov) == 1):
                ou, ov = np.repeat(ou, na), np.repeat(ov, na)
            if len(oa) == 1:
                oa = np.repeat(oa, na)
            detector_array:np.ndarray = deepcopy(projectConeBeamGPU(detector_array, object_array, transformationMatrix, nx, ny, nz, nu, nv, nw, na, su, sv, ou, ov, oa, s2d, near, far))
        else:
            detector_array:np.ndarray = deepcopy(projectParallelBeamGPU(detector_array, object_array, transformationMatrix, nx, ny, nz, nu, nv, nw, na))
        detector_array = detector_array.reshape(na, nv, nu)
    
    else:
        detector_array = np.zeros([na, nv, nu])
        if mode:
            ou, ov = parameters.detector.motion.translation.get(axis=0).astype(np.float32)
            oa = parameters.detector.motion.rotation.get().astype(np.float32)
            if (len(ou) == 1) and (len(ov) == 1):
                ou, ov = np.repeat(ou, na), np.repeat(ov, na)
            if len(oa) == 1:
                oa = np.repeat(oa, na)
            projectConeBeamCPU(detector_array, object_array, transformationMatrix, nx, ny, nz, nu, nv, nw, na, su, sv, ou, ov, oa, s2d, near, far)
        else:
            projectParallelBeamCPU(detector_array, object_array, transformationMatrix, nx, ny, nz, nu, nv, nw, na)
    
    return detector_array * ray_step


def _getNearFar(params:_Parameters, ray_step:float):
    if ray_step < 0:
        raise ValueError("ray_step must be larger than zero.")
    s2o = params.source.distance.source2origin
    s2d = params.source.distance.source2detector
    lo:float = np.linalg.norm(params.object.length.get()/2)
    to:float = np.linalg.norm(params.object.motion.translation.get()-params.source.motion.translation.get(), axis=1).max()
    td:float = params.detector.length.get()/2 + np.abs(params.detector.motion.translation.get())
    near = max(0., s2o-lo-to)
    far = min(s2o+lo+to, np.sqrt(s2d**2 + np.sum(td**2, axis=1).max())) if params.mode else min(s2o+lo+to, s2d)
    if ray_step > 0:
        nw = int((far - near) / ray_step)
        far = near + nw*ray_step
        return near, far, nw
    else:
        return near, far