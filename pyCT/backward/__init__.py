import pyCT
from pyCT.parameter import _Parameters
from .reconstructionCPU import *
from reconstructionGPU import reconstructParallelBeamGPU, reconstructConeBeamGPU
from copy import deepcopy

def reconstruct(sinogram_array : np.ndarray,
                parameters : _Parameters,
                filter : str|None = 'ramp',
                **kwargs):
    '''
    key : cuda, offset
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
    
    # set offset correction
    if 'offset' in kwargs.keys():
        is_offsetCorrection:bool = kwargs['offset']
    else:
        is_offsetCorrection:bool = False

    # check filter
    FILTERS = ['none', 'ramp', 'ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann']
    assert (filter is None) or filter.lower() in FILTERS, "{} was not supported in pyCT'.format(filter) + '\nWe support the following filters: ramp or ram-lak, shepp-logan, cosine, hamming, hann"

    # get parameters
    mode = parameters.mode
    s2d = parameters.source.distance.source2detector
    nx, ny, nz = parameters.object.size.get()
    du, dv = parameters.detector.spacing.get()
    su, sv = parameters.detector.length.get()
    nu, nv = parameters.detector.size.get()
    na = len(parameters.source.motion.rotation.get()[0])

    # get transformation
    transformation = pyCT.getTransformation(parameters, 1, 0, s2d)
    transformationMatrix = transformation.getBackward()

    if is_offsetCorrection:
        sinogram_array = _applyOffsetCorrection(sinogram_array, parameters)
    
    if filter is None or filter.lower() == 'none':
        pass
    else:
        sinogram_array = _applyFilter(sinogram_array, parameters, filter)

    if is_cuda:
        reconstruction_array = np.zeros(nz*ny*nx, dtype=np.float32)
        sinogram_array = sinogram_array.flatten().astype(np.float32)
        transformationMatrix = transformationMatrix.flatten().astype(np.float32)
        if mode:
            ou, ov = parameters.detector.motion.translation.get(axis=0).astype(np.float32)
            oa = parameters.detector.motion.rotation.get().astype(np.float32)
            if (len(ou) == 1) and (len(ov) == 1):
                ou, ov = np.repeat(ou, na), np.repeat(ov, na)
            if len(oa) == 1:
                oa = np.repeat(oa, na)
            reconstruction_array:np.ndarray = deepcopy(reconstructConeBeamGPU(reconstruction_array, sinogram_array, transformationMatrix, nx, ny, nz, nu, nv, na, su, sv, du, dv, ou, ov, oa, s2d))
        else:
            reconstruction_array:np.ndarray = deepcopy(reconstructParallelBeamGPU(reconstruction_array, sinogram_array, transformationMatrix, nx, ny, nz, nu, nv, na))
        reconstruction_array = reconstruction_array.reshape(nz, ny, nx)
    else:
        reconstruction_array = np.zeros([nz, ny, nx])
        if mode:
            reconstructConeBeamCPU(reconstruction_array, sinogram_array, transformationMatrix, nx, ny, nz, nu, nv, na, su, sv, du, dv, ou, ov, oa, s2d)
        else:
            reconstructParallelBeamCPU(reconstruction_array, sinogram_array, transformationMatrix, nx, ny, nz, nu, nv, na)
    
    return reconstruction_array


def _applyOffsetCorrection(sinogram_array : np.ndarray, 
                           parameters : _Parameters):
    weight = np.ones(sinogram_array.shape)
    ou, _ = parameters.detector.motion.translation.get().T
    oa = parameters.detector.motion.rotation.get()
    if np.any(ou-ou[0]) or np.any(oa):
        raise ValueError()
    else:
        ou = ou[0]
    gap = int((parameters.detector.length.u/2 - ou) / parameters.detector.spacing.u)
    if ou != 0 and gap > 0:
        if ou > 0:
            f = (1+np.cos(np.linspace(-np.pi, 0, gap*2))) / 2
            weight[:,:,:2*gap] = f
        elif ou < 0:
            f =  (1+np.cos(np.linspace(0, np.pi, gap*2))) / 2
            weight[:,:,-2*gap:] = f
        return 2 * weight * sinogram_array
    else:
        return sinogram_array

def _applyFilter(sinogram_array : np.ndarray, 
                 parameters : _Parameters, 
                 filter : str):
    na = len(parameters.source.motion.rotation.get()[0])
    nu = parameters.detector.size.u
    du = parameters.detector.spacing.u
    s2o = parameters.source.distance.source2origin
    s2d = parameters.source.distance.source2detector

    extended_size = max(64, int(2 ** np.ceil(np.log2(2 * nu))))
    pad = (extended_size - nu) // 2

    n = np.concatenate((np.arange(1, extended_size/2 + 1, 2, dtype=np.uint32),
                            np.arange(extended_size/2 - 1, 0, -2, dtype=np.uint32)))
    f = np.zeros(extended_size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    fourier_filter = 2 * np.real(np.fft.fft(f))
    fourier_proj = np.fft.fft(np.pad(sinogram_array, [(0,0),(0,0),(pad,pad)]), axis=-1)

    if filter.lower() == 'ramp' or filter.lower()=='ram-lak':
        pass
    elif filter.lower() == 'shepp-logan':
        omega = np.pi * np.fft.fftfreq(extended_size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter.lower() == 'cosine':
        freq = np.linspace(0, np.pi, extended_size, endpoint=False)
        cosine_filter = np.fft.fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter.lower() == 'hamming':
        fourier_filter *= np.fft.fftshift(np.hamming(extended_size))
    elif filter.lower() == 'hann':
        fourier_filter *= np.fft.fftshift(np.hanning(extended_size))

    sinogram_array = np.fft.ifft(fourier_proj*fourier_filter, axis=-1).real
    if parameters.mode:
        weight = np.pi / na / 2 / du / s2o * s2d
    else:
        weight = np.pi / na / 2 / du
    return sinogram_array[..., pad : -pad] * weight