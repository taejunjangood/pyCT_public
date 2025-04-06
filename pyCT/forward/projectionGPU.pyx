import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef cnp.float32_t DTYPE

cdef extern from "forward.h":
    void funcParallelBeam(float* detector_array, float* object_array, float* transformation, int nx, int ny, int nz, int nu, int nv, int nw, int na)
    void funcConeBeam    (float* detector_array, float* object_array, float* transformation, int nx, int ny, int nz, int nu, int nv, int nw, int na, float su, float sv, float* ou, float* ov, float* oa, float s2d, float near, float far);


def projectParallelBeamGPU(cnp.ndarray[DTYPE, ndim=1] detector_array, cnp.ndarray[DTYPE, ndim=1] object_array, cnp.ndarray[DTYPE, ndim=1] transformation, int nx, int ny, int nz, int nu, int nv, int nw, int na):

    cdef float* c_detector_array = <float *> detector_array.data
    cdef float* c_object_array = <float *> object_array.data
    cdef float* c_transformation = <float *> transformation.data

    funcParallelBeam(c_detector_array, c_object_array, c_transformation, nx, ny, nz, nu, nv, nw, na)

    cdef cnp.npy_intp shape[1]
    shape[0] = <cnp.npy_intp> (na*nv*nu)

    new = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_FLOAT32, c_detector_array)

    return new


def projectConeBeamGPU(cnp.ndarray[DTYPE, ndim=1] detector_array, cnp.ndarray[DTYPE, ndim=1] object_array, cnp.ndarray[DTYPE, ndim=1] transformation, int nx, int ny, int nz, int nu, int nv, int nw, int na, float su, float sv, cnp.ndarray[DTYPE, ndim=1] ou, cnp.ndarray[DTYPE, ndim=1] ov, cnp.ndarray[DTYPE, ndim=1] oa, float s2d, float near, float far):

    cdef float* c_detector_array = <float *> detector_array.data
    cdef float* c_object_array = <float *> object_array.data
    cdef float* c_transformation = <float *> transformation.data
    cdef float* c_ou = <float *> ou.data
    cdef float* c_ov = <float *> ov.data
    cdef float* c_oa = <float *> oa.data

    funcConeBeam(c_detector_array, c_object_array, c_transformation, nx, ny, nz, nu, nv, nw, na, su, sv, c_ou, c_ov, c_oa, s2d, near, far)

    cdef cnp.npy_intp shape[1]
    shape[0] = <cnp.npy_intp> (na*nv*nu)

    new = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_FLOAT32, c_detector_array)

    return new