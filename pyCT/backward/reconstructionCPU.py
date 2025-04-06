import numpy as np

def reconstructParallelBeamCPU(reconstruction_array, sinogram_array, transformation, nx, ny, nz, nu, nv, na):
    pad = 1
    sinogram_array = np.pad(sinogram_array, [(0,0),(pad,pad),(pad,pad)])
    nu += (2*pad -1)
    nv += (2*pad -1)
    for a in range(na):
        matrix = transformation[a]
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    u, v, _, _ = matrix @ [x,y,z,1] + pad
                    if 0 < u < nu and 0 < v < nv:
                        u0, v0 = int(u), int(v)
                        u1, v1 = u0 + 1, v0 + 1
                        interp = (v1-v) * ((u1-u)*sinogram_array[a,v0,u0]+(u-u0)*sinogram_array[a,v0,u1]) + (v-v0) * ((u1-u)*sinogram_array[a,v1,u0]+(u-u0)*sinogram_array[a,v1,u1])
                        reconstruction_array[z,y,x] += interp


def reconstructConeBeamCPU(reconstruction_array, sinogram_array, transformation, nx, ny, nz, nu, nv, na, su, sv, du, dv, ou, ov, oa, s2d):
    pad = 1
    sinogram_array = np.pad(sinogram_array, [(0,0),(pad,pad),(pad,pad)])
    nu += (2*pad -1)
    nv += (2*pad -1)
    if len(ou) == 1:
        ou = np.repeat(ou, na)
    if len(ov) == 1:
        ov = np.repeat(ov, na)
    if len(oa) == 1:
        oa = np.repeat(oa, na)
    for a in range(na):
        matrix = transformation[a]
        offmat = [[np.cos(oa[a]), np.sin(oa[a])],[-np.sin(oa[a]), np.cos(oa[a])]]
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    u, v, w, _ = matrix @ [x,y,z,1]
                    u, v = ([u, v] @ offmat / w * -s2d - [ou[a], ov[a]]) / [du, dv] + pad
                    if 0 < u < nu and 0 < v < nv:
                        u0, v0 = int(u), int(v)
                        u1, v1 = u0 + 1, v0 + 1
                        interp = (v1-v) * ((u1-u)*sinogram_array[a,v0,u0]+(u-u0)*sinogram_array[a,v0,u1]) + (v-v0) * ((u1-u)*sinogram_array[a,v1,u0]+(u-u0)*sinogram_array[a,v1,u1])
                        reconstruction_array[z,y,x] += interp