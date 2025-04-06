import numpy as np

def projectParallelBeamCPU(detector_array, object_array, transformation, nx, ny, nz, nu, nv, nw, na):
    pad = 1
    object_array = np.pad(object_array, pad)
    nx += (2*pad -1)
    ny += (2*pad -1)
    nz += (2*pad -1)    
    for a in range(na):
        matrix = transformation[a]
        for u in range(nu):
            for v in range(nv):
                for w in range(nw):
                    x, y, z, _ = matrix @ [u,v,w,1] + pad
                    if 0 < x < nx and 0 < y < ny and 0 < z < nz:
                        x0, y0, z0 = int(x), int(y), int(z)
                        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
                        interp = (z1 - z) * ((y1 - y) * ((x1 - x) * object_array[z0, y0, x0] + (x - x0) * object_array[z0, y0, x1]) + (y - y0) * ((x1 - x) * object_array[z0, y1, x0] + (x - x0) * object_array[z0, y1, x1])) + (z - z0) * ((y1 - y) * ((x1 - x) * object_array[z1, y0, x0] + (x - x0) * object_array[z1, y0, x1]) + (y - y0) * ((x1 - x) * object_array[z1, y1, x0] + (x - x0) * object_array[z1, y1, x1]))
                        detector_array[a,v,u] += interp


def projectConeBeamCPU(detector_array, object_array, transformation, nx, ny, nz, nu, nv, nw, na, su, sv, ou, ov, oa, s2d, near, far):
    pad = 1
    object_array = np.pad(object_array, pad)
    nx += (2*pad -1)
    ny += (2*pad -1)
    nz += (2*pad -1)
    U, V = np.meshgrid(np.linspace(-su/2+(su/nu)/2, su/2-(su/nu)/2, nu), np.linspace(-sv/2+(sv/nv)/2, sv/2-(sv/nv)/2, nv))
    U, V = np.stack([U.flatten(), V.flatten()], axis=1) @ [[np.cos(oa), -np.sin(oa)], [np.sin(oa), np.cos(oa)]] + np.stack([ou[None], ov[None]])
    U, V = U.reshape(nv, nu, na).transpose(2,0,1), V.reshape(nv, nu, na).transpose(2,0,1)
    W = -s2d * np.ones_like(U)
    directions = np.stack([U,V,W], axis=-1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    for a in range(na):
        matrix = transformation[a]
        for u in range(nu):
            for v in range(nv):
                direction = directions[a,v,u]
                for w in range(nw):
                    x, y, z = (near + w*(far-near)/nw) * direction
                    x, y, z, _ = matrix @ [x,y,z,1] + pad
                    if 0 < x < nx and 0 < y < ny and 0 < z < nz:
                        x0, y0, z0 = int(x), int(y), int(z)
                        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
                        interp = (z1 - z) * ((y1 - y) * ((x1 - x) * object_array[z0, y0, x0] + (x - x0) * object_array[z0, y0, x1]) + (y - y0) * ((x1 - x) * object_array[z0, y1, x0] + (x - x0) * object_array[z0, y1, x1])) + (z - z0) * ((y1 - y) * ((x1 - x) * object_array[z1, y0, x0] + (x - x0) * object_array[z1, y0, x1]) + (y - y0) * ((x1 - x) * object_array[z1, y1, x0] + (x - x0) * object_array[z1, y1, x1]))
                        detector_array[a,v,u] += interp