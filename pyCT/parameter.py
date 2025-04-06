import json
import numpy as np
from copy import deepcopy

def getParameters(path_header:str|None = None):
    return _Parameters(path_header)

class _Parameters():
    def __init__(self, path_params:str):
        self.mode:bool = None
        self.object = _Object()
        self.source = _Source()
        self.detector = _Detector()

        if path_params is not None:
            self.load(path_params)
        
    def __str__(self):
        output = ''
        if self.mode:
            output += 'mode                    : cone\n'
        else:
            output += 'mode                    : parallel\n'
        output +=     'object size     (voxel) : ({}, {} ,{})\n'.format(self.object.size.x, self.object.size.y, self.object.size.z)
        output +=     'object spacing     (mm) : ({}, {} ,{})\n'.format(self.object.spacing.x, self.object.spacing.y, self.object.spacing.z)
        output +=     'object length      (mm) : ({}, {} ,{})\n'.format(self.object.length.x, self.object.length.y, self.object.length.z)
        output +=     'detector size   (pixel) : ({}, {})\n'.format(self.detector.size.u, self.detector.size.v)
        output +=     'detector spacing   (mm) : ({}, {})\n'.format(self.detector.spacing.u, self.detector.spacing.v)
        output +=     'detector length    (mm) : ({}, {})\n'.format(self.detector.length.u, self.detector.length.v)
        output +=     'source to origin   (mm) : {}\n'.format(self.source.distance.source2origin)
        output +=     'source to detector (mm) : {}\n'.format(self.source.distance.source2detector)
        return output
    
    def load(self, path:str):
        meta = json.load(open(path, 'r'))
        
        self.mode = meta['mode']
        # object
        self.object.size.set(list(meta['object']['size'].values()))
        self.object.spacing.set(list(meta['object']['spacing'].values()))
        self.object.length.set(list(meta['object']['length'].values()))
        self.object.motion.rotation.angles = meta['object']['motion']['rotation']['angles']
        self.object.motion.rotation.axes = meta['object']['motion']['rotation']['axes']
        self.object.motion.translation.vectors = meta['object']['motion']['translation']['vectors']
        # source
        self.source.distance.source2origin = meta['source']['distance']['source2origin']
        self.source.distance.source2detector = meta['source']['distance']['source2detector']
        self.source.motion.rotation.angles = meta['source']['motion']['rotation']['angles']
        self.source.motion.rotation.axes = meta['source']['motion']['rotation']['axes']
        self.source.motion.translation.vectors = meta['source']['motion']['translation']['vectors']
        # detector
        self.detector.size.set(list(meta['detector']['size'].values()))
        self.detector.spacing.set(list(meta['detector']['spacing'].values()))
        self.detector.length.set(list(meta['detector']['length'].values()))
        self.detector.motion.rotation.angles = meta['detector']['motion']['rotation']['angles']
        self.detector.motion.translation.vectors = meta['detector']['motion']['translation']['vectors']

    def save(self, path:str):
        meta = deepcopy(self.__dict__)
        stack = [meta]
        while stack:
            element = stack.pop()
            for key, value in element.items():
                if type(value) in [bool, int, float, str, list, tuple]:
                    pass
                elif type(value) == np.ndarray:
                    element[key] = value.tolist()
                elif type(value) in [np.float128,np.float64, np.float32, np.float16]:
                    element[key] = float(value)
                elif type(value) in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8]:
                    element[key] = int(value)
                else:
                    element[key] = value.__dict__
                    stack.append(element[key])
        json.dump(meta, open(path, 'w'), indent=2)

    def check(self):
        for key, value in {'object':self.object, 'detector':self.detector}.items():
            l = [value.size.checkNone(), value.spacing.checkNone(), value.length.checkNone()]
            if sum(l) == 0:
                assert not any(value.length.get() / value.size.get() - value.spacing.get()), "{0} : The relation between length, spacing, and size is incorrect.".format(key)
            elif sum(l) == 1:
                if l[0]:
                    value.size.set((value.length.get() / value.spacing.get()).astype(np.uint64))
                elif l[1]:
                    value.spacing.set(value.length.get() / value.size.get())
                else:
                    value.length.set(value.size.get() * value.spacing.get())
            else:
                assert False, "{0} : At least two inputs are required: size, spacing, and length.".format(key)
        
        assert not self.source.distance.checkNone(), "Two inputs are required: source2origin and source2detector."
        assert self.source.distance.source2origin < self.source.distance.source2detector, "source2detector must be larger than source2origin."

    def set(self, source_angles=[0], detector_offset=[0,0]):
        self.source.motion.rotation.set(source_angles, axes='z')
        self.detector.motion.translation.set(detector_offset[0], detector_offset[1])
        
    def copy(self):
        return deepcopy(self)
        

class _2D():
    def __init__(self, u:float|int=None, v:float|int=None):
        self.u = u
        self.v = v
    def set(self, ls:list | tuple | np.ndarray):
        self.u = ls[0]
        self.v = ls[1]
    def get(self):
        return np.array([self.u, self.v])
    def checkNone(self) -> bool:
        if (self.u == None) or (self.v == None):
            return True
        else:
            return False

class _3D():
    def __init__(self, x:float|int=None, y:float|int=None, z:float|int=None):
        self.x = x
        self.y = y
        self.z = z
    def set(self, ls:list | tuple | np.ndarray):
        self.x = ls[0]
        self.y = ls[1]
        self.z = ls[2]
    def get(self):
        return np.array([self.x, self.y, self.z])
    def checkNone(self) -> bool:
        if (self.x == None) or (self.y == None) or (self.z == None):
            return True
        else:
            return False

class _Rotation2D():
    def __init__(self):
        self.angles:list = [0]
    
    def get(self):
        return np.array(self.angles)
    
    def set(self, angles):
        if type(angles) in TYPE:
            angles = [float(angles)]
        elif type(angles) == tuple:
            angles = list(angles)
        elif type(angles) == np.ndarray:
            angles = angles.tolist()
        self.angles = angles

class _Rotation3D():
    def __init__(self):
        self.angles:list = [[0]]
        self.axes:str = 'z'
    
    def get(self, axis=1):
        return np.stack(self.angles, axis=axis), self.axes
    
    def set(self, *angles:list|np.ndarray, axes:str):
        if len(angles) != len(axes):
            raise ValueError("The number of angles ({0}) and the number of axes ({1}) do not match.".format(len(angles), len(axes)))
        angles = list(angles)
        for i, angle in enumerate(angles):
            if type(angle) in TYPE:
                angles[i] = [float(angle)]
            elif type(angle) == tuple:
                angles[i] = list(angle)
            elif type(angle) == np.ndarray:
                angles[i] = angle.tolist()
        if all([len(angle)-len(angles[0]) for angle in angles]):
            raise ValueError("The numbers of input angles do not match.")
        for axis in axes:
            if axis.lower() not in ['x', 'y', 'z']:
                raise ValueError("The input axis ({}) does not contain \{x,y,z\}".format(axis))
        self.angles = angles
        self.axes = axes    

class _Translation():
    def __init__(self, dim:int):
        self.vectors:list = [[0]]*dim
        self.dim:int = dim
    
    def get(self, axis:int=1):
        return np.stack(self.vectors, axis=axis)
    
    def set(self, *vectors:np.ndarray):
        vectors = list(vectors)
        for i, vector in enumerate(vectors):
            if type(vector) in TYPE:
                vectors[i] = ([float(vector)])
            elif type(vector) == tuple:
                vectors[i] = list(vector)
            elif type(vector) == np.ndarray:
                vectors[i] = vector.tolist()
        if len(vectors) != self.dim:
            raise ValueError("The input vectors are not {}D vectors.".format(self.dim))
        if all([len(vector)-len(vectors[0]) for vector in vectors]):
            raise ValueError("The numbers of input vectors do not match.")
        self.vectors = vectors

class _Motion():
    def __init__(self, dim:int):
        if dim == 2:
            self.rotation = _Rotation2D()
        elif dim == 3:
            self.rotation = _Rotation3D()            
        else:
            raise ValueError("Only 2D and 3D are supported.")
        self.translation = _Translation(dim)

    def checkSize(self) -> bool:
        if (len(self.rotation.angles) != len(self.translation.vectors)) and (len(self.translation.vectors) != 1):
            return False
        else:
            return True

class _Distance():
    def __init__(self):
        self.source2origin:float = None
        self.source2detector:float = None
    def checkNone(self) -> bool:
        return (self.source2origin is None) or (self.source2detector is None)

class _Object():
    def __init__(self):
        self.size = _3D()
        self.length = _3D()
        self.spacing = _3D()
        self.motion = _Motion(dim=3)
        
class _Source():
    def __init__(self):
        self.distance = _Distance()
        self.motion = _Motion(dim=3)

class _Detector():
    def __init__(self):
        self.size = _2D()
        self.length = _2D()
        self.spacing = _2D()
        self.motion = _Motion(dim=2)

TYPE = [int, float, 
        np.int8, np.int16, np.int32, np.int64, 
        np.uint8, np.uint16, np.uint32, np.uint64, 
        np.float16, np.float32, np.float64, np.float128]