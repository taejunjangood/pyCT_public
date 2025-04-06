import numpy as np
from pyCT.parameter import _Parameters

def getTransformation(params, nw, near, far):
    return _Transformation(params, nw, near, far)

class _Transformation():
    def __init__(self, params:_Parameters, nw:int, near:float, far:float):
        self.worldTransformation  : np.ndarray  = None
        self.cameraTransformation : np.ndarray  = None
        self.viewTransformation   : np.ndarray  = None
        self.__params             : _Parameters = params

        self.__setWorldTransformation()
        self.__setCameraTransformation()
        if not params.mode:
            self.__setViewTransformation(nw, near, far)

    def getForward(self) -> np.ndarray:
        return np.linalg.inv(self.getBackward())
    
    def getBackward(self) -> np.ndarray:
        if self.__params.mode:
            return np.einsum('aij,ajk->aik', self.cameraTransformation, self.worldTransformation)
        else:
            return self.viewTransformation @ np.einsum('aij,ajk->aik', self.cameraTransformation, self.worldTransformation)


    def __setWorldTransformation(self):
        dx, dy, dz = self.__params.object.spacing.get()
        sx, sy, sz = self.__params.object.length.get()
        angles, axes = self.__params.object.motion.rotation.get()
        vectors = self.__params.object.motion.translation.get()
        self.worldTransformation = np.array([[dx, 0 , 0 , -sx/2+dx/2], 
                                             [0 , dy, 0 , -sy/2+dy/2], 
                                             [0 , 0 , dz, -sz/2+dz/2], 
                                             [0 , 0 , 0 , 1]])
        # [A,4,4]x[4,4] -> [A,4,4]
        self.worldTransformation = np.einsum('aij,jk->aik', _getRotation(angles, axes), self.worldTransformation)
        self.worldTransformation = np.einsum('aij,ajk->aik', _getTranslation(vectors), self.worldTransformation)

    
    def __setCameraTransformation(self):
        angles, axes = self.__params.source.motion.rotation.get()
        vectors = self.__params.source.motion.translation.get()
        s2o = self.__params.source.distance.source2origin
        detectorFrame = np.array([[0,0,1,0],
                                  [1,0,0,0],
                                  [0,1,0,0],
                                  [0,0,0,1]])
        # [A,4,4]x[4,4] -> [A,4,4]
        detectorFrame = np.einsum('aij,jk->aki', _getRotation(angles, axes), detectorFrame)
        sourceOrigin = s2o * detectorFrame[:, 2, :3] + vectors # [A,3]
        self.cameraTransformation = np.einsum('aij,ajk->aik', detectorFrame, _getTranslation(-sourceOrigin))


    def __setViewTransformation(self, nw:int, near:float, far:float):
        if self.__params.mode:
            pass
        else:
            nu, nv = self.__params.detector.size.get()
            du, dv = self.__params.detector.spacing.get()
            angles = self.__params.detector.motion.rotation.get()
            vectors = self.__params.detector.motion.translation.get()
            length = far - near
            viewMatrix = np.array(
                [
                    [1/du, 0   , 0         , -1/2+nu/2],
                    [0   , 1/dv, 0         , -1/2+nv/2],
                    [0   , 0   , -nw/length, -nw*near/length],
                    [0   , 0   , 0         , 1]
                ]
            )
            motionMatrix = _getRotation(angles, 'z') @ _getTranslation(-1*vectors)
            self.viewTransformation = np.einsum('ij,ajk -> aik', viewMatrix, motionMatrix)


def _makeRotation(angle, 
                  axis:str
                  ):
    '''
    angle : (1,)
    axis  : (1,)
    ->
    mat    : (4,4)
    '''
    if type(angle) in [int, float, 
                       np.uint64, np.uint32, np.uint16, np.uint8, np.int64, np. int32, np.int16, np.int8,
                       np.float128, np.float64, np.float32, np.float16]:
        if axis == 'z':
            return np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                             [np.sin(angle),  np.cos(angle), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        elif axis == 'y':
            return np.array([[np.cos(angle), 0, -np.sin(angle), 0],
                             [0, 1, 0, 0],
                             [np.sin(angle), 0, np.cos(angle), 0],
                             [0, 0, 0, 1]])
        elif axis == 'x':
            return np.array([[1, 0, 0, 0],
                             [0, np.cos(angle), -np.sin(angle), 0],
                             [0, np.sin(angle), np.cos(angle), 0],
                             [0, 0, 0, 1]])
        
    elif type(angle) in [list, tuple, np.ndarray]:
        na = len(angle)
        if axis == 'z':
            return np.array([[np.cos(angle), -np.sin(angle), np.zeros(na), np.zeros(na)],
                             [np.sin(angle),  np.cos(angle), np.zeros(na), np.zeros(na)],
                             [np.zeros(na), np.zeros(na), np.ones(na), np.zeros(na)],
                             [np.zeros(na), np.zeros(na), np.zeros(na), np.ones(na)]]).transpose(2,0,1)
        elif axis == 'y':
            return np.array([[np.cos(angle), np.zeros(na), -np.sin(angle), np.zeros(na)],
                             [np.zeros(na), np.ones(na), np.zeros(na), np.zeros(na)],
                             [np.sin(angle), np.zeros(na), np.cos(angle), np.zeros(na)],
                             [np.zeros(na), np.zeros(na), np.zeros(na), np.ones(na)]]).transpose(2,0,1)
        elif axis == 'x':
            return np.array([[np.ones(na), np.zeros(na), np.zeros(na), np.zeros(na)],
                             [np.zeros(na), np.cos(angle), -np.sin(angle), np.zeros(na)],
                             [np.zeros(na), np.sin(angle), np.cos(angle), np.zeros(na)],
                             [np.zeros(na), np.zeros(na), np.zeros(na), np.ones(na)]]).transpose(2,0,1)
        else:
            assert False, "axis must be entered by one in {x, y, z}."

def _getRotation(angles:np.ndarray, axes:str):
    '''
    angles : (num_angles, num_rots)
    axes   : (num_rots)
    ->
    output : (num_angles, 4, 4)
    '''
    # angles: [na, nc], axes: [nc] >> out: [na,4,4]
    na = len(angles)
    R = np.eye(4)[None].repeat(na, axis=0)
    # (nc) loops
    for angle, axis in zip(angles.T, axes):
        R = _makeRotation(angle, axis) @ R
    return R
    
def _getTranslation(offset:np.ndarray):
    na, dim = offset.shape
    assert (dim == 3) or (dim == 2), "Dimension must be 2 or 3. "
    
    R = np.eye(4)[None,...].repeat(na, axis=0)
    R[:, :dim, -1] = offset
    return R