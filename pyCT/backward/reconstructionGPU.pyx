import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef cnp.float32_t DTYPE

cdef extern from "backward.h":
    void funcParallelBeam(float* reconstruction_array, float* sinogram_array, float* transformation, int nx, int ny, int nz, int nu, int nv, int na)
    void funcConeBeam    (float* reconstruction_array, float* sinogram_array, float* transformation, int nx, int ny, int nz, int nu, int nv, int na, float su, float sv, float du, float dv, float* ou, float* ov, float* oa, float s2d);

def reconstructParallelBeamGPU(cnp.ndarray[DTYPE, ndim=1] reconstruction_array, cnp.ndarray[DTYPE, ndim=1] sinogram_array, cnp.ndarray[DTYPE, ndim=1] transformation, int nx, int ny, int nz, int nu, int nv, int na):

    cdef float* c_reconstruction_array = <float *> reconstruction_array.data
    cdef float* c_sinogram_array = <float *> sinogram_array.data
    cdef float* c_transformation = <float *> transformation.data

    funcParallelBeam(c_reconstruction_array, c_sinogram_array, c_transformation, nx, ny, nz, nu, nv, na)

    cdef cnp.npy_intp shape[1]
    shape[0] = <cnp.npy_intp> (nx*ny*nz)

    new = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_FLOAT32, c_reconstruction_array)

    return new

def reconstructConeBeamGPU(cnp.ndarray[DTYPE, ndim=1] reconstruction_array, cnp.ndarray[DTYPE, ndim=1] sinogram_array, cnp.ndarray[DTYPE, ndim=1] transformation, int nx, int ny, int nz, int nu, int nv, int na, float su, float sv, float du, float dv, cnp.ndarray[DTYPE, ndim=1] ou, cnp.ndarray[DTYPE, ndim=1] ov, cnp.ndarray[DTYPE, ndim=1] oa, float s2d):

    cdef float* c_reconstruction_array = <float *> reconstruction_array.data
    cdef float* c_sinogram_array = <float *> sinogram_array.data
    cdef float* c_transformation = <float *> transformation.data
    cdef float* c_ou = <float *> ou.data
    cdef float* c_ov = <float *> ov.data
    cdef float* c_oa = <float *> oa.data

    funcConeBeam(c_reconstruction_array, c_sinogram_array, c_transformation, nx, ny, nz, nu, nv, na, su, sv, du, dv, c_ou, c_ov, c_oa, s2d)

    cdef cnp.npy_intp shape[1]
    shape[0] = <cnp.npy_intp> (nx*ny*nz)

    new = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_FLOAT32, c_reconstruction_array)

    return new