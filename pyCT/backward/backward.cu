#include "backward.h"

__global__ 
void kernel_parallel(float* recon, cudaTextureObject_t texObjSino, float* transformation, int na)
{
	int nx = gridDim.x;
	int ny = gridDim.y;
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;

	float t00, t01, t02, t03;
	float t10, t11, t12, t13;
	float t20, t21, t22, t23;
	float u, v;
	int idx;

	for (int a = 0; a < na; a++)
	{
		t00 = transformation[0 + 0*4 + a*4*4];
		t01 = transformation[1 + 0*4 + a*4*4];
		t02 = transformation[2 + 0*4 + a*4*4];
		t03 = transformation[3 + 0*4 + a*4*4];

		t10 = transformation[0 + 1*4 + a*4*4];
		t11 = transformation[1 + 1*4 + a*4*4];
		t12 = transformation[2 + 1*4 + a*4*4];
		t13 = transformation[3 + 1*4 + a*4*4];

		u = t00 * x + t01 * y + t02 * z + t03;
		v = t10 * x + t11 * y + t12 * z + t13;

		idx = x + y*nx + z*nx*ny;
		recon[idx] += tex3D<float>(texObjSino, u+.5, v+.5, a+.5);
	}
}

void funcParallelBeam(float* reconstruction_array, float* sinogram_array, float* transformation, int nx, int ny, int nz, int nu, int nv, int na)
{
	// object array >> texture memory
	const cudaExtent objSize = make_cudaExtent(nu, nv, na);
	cudaArray* d_sinogram_array = 0;
	cudaTextureObject_t tex_sinogram_array = 0;

	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMalloc3DArray(&d_sinogram_array, &channelDesc, objSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)sinogram_array, objSize.width * sizeof(float), objSize.width, objSize.height);
	copyParams.dstArray = d_sinogram_array;
	copyParams.extent = objSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_sinogram_array;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = false; // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear; // linear interpolation
	texDescr.addressMode[0] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.addressMode[1] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.addressMode[2] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&tex_sinogram_array, &texRes, &texDescr, NULL);

	float* d_transformation;
	cudaMalloc(&d_transformation, na * 4 * 4 * sizeof(float));
	cudaMemcpy(d_transformation, transformation, na * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
	float* d_reconstruction_array;
	cudaMalloc(&d_reconstruction_array, nx * ny * nz * sizeof(float));
	
	kernel_parallel <<< dim3(nx,ny,1), dim3(nz,1,1) >>> (d_reconstruction_array, tex_sinogram_array, d_transformation, na);
	cudaMemcpy(reconstruction_array, d_reconstruction_array, nx*ny*nz*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_reconstruction_array);
	cudaFree(d_transformation);
	cudaFreeArray(d_sinogram_array);
	cudaDestroyTextureObject(tex_sinogram_array);
}

__global__ 
void kernel_cone(float* recon, cudaTextureObject_t texObjSino, float* transformation, int na, float su, float sv, float du, float dv, float *ou, float *ov, float *oa, float s2d)
{
	int nx = gridDim.x;
	int ny = gridDim.y;
	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;
	
	float t00, t01, t02, t03;
	float t10, t11, t12, t13;
	float t20, t21, t22, t23;
	float u, v, w;
	float u_, v_;
	int idx;

	for (int a = 0; a < na; a++)
	{
		t00 = transformation[0 + 0*4 + a*4*4];
		t01 = transformation[1 + 0*4 + a*4*4];
		t02 = transformation[2 + 0*4 + a*4*4];
		t03 = transformation[3 + 0*4 + a*4*4];

		t10 = transformation[0 + 1*4 + a*4*4];
		t11 = transformation[1 + 1*4 + a*4*4];
		t12 = transformation[2 + 1*4 + a*4*4];
		t13 = transformation[3 + 1*4 + a*4*4];

		t20 = transformation[0 + 2*4 + a*4*4];
		t21 = transformation[1 + 2*4 + a*4*4];
		t22 = transformation[2 + 2*4 + a*4*4];
		t23 = transformation[3 + 2*4 + a*4*4];
		
		u = t00 * x + t01 * y + t02 * z + t03;
		v = t10 * x + t11 * y + t12 * z + t13;
		w = t20 * x + t21 * y + t22 * z + t23;

		u_ = cosf(oa[a])*u - sinf(oa[a])*v;
		v_ = sinf(oa[a])*u + cosf(oa[a])*v;
		u = (u_ / w * -s2d + su/2 - ou[a])/du;
		v = (v_ / w * -s2d + sv/2 - ov[a])/dv;

		idx = x + y*nx + z*nx*ny;
		recon[idx] += tex3D<float>(texObjSino, u+.5, v+.5, a+.5);
	}
}

void funcConeBeam(float* reconstruction_array, float* sinogram_array, float* transformation, int nx, int ny, int nz, int nu, int nv, int na, float su, float sv, float du, float dv, float* ou, float* ov, float* oa, float s2d)
{
	// object array >> texture memory
	const cudaExtent objSize = make_cudaExtent(nu, nv, na);
	cudaArray* d_sinogram_array = 0;
	cudaTextureObject_t tex_sinogram_array = 0;

	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMalloc3DArray(&d_sinogram_array, &channelDesc, objSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)sinogram_array, objSize.width * sizeof(float), objSize.width, objSize.height);
	copyParams.dstArray = d_sinogram_array;
	copyParams.extent = objSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_sinogram_array;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = false; // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear; // linear interpolation
	texDescr.addressMode[0] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.addressMode[1] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.addressMode[2] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&tex_sinogram_array, &texRes, &texDescr, NULL);
	
	float* d_transformation;
	cudaMalloc(&d_transformation, na * 4 * 4 * sizeof(float));
	cudaMemcpy(d_transformation, transformation, na * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
	float* d_reconstruction_array;
	cudaMalloc(&d_reconstruction_array, nx * ny * nz * sizeof(float));
	float* d_ou;
	cudaMalloc(&d_ou, na * sizeof(float));
	cudaMemcpy(d_ou, ou, na * sizeof(float), cudaMemcpyHostToDevice);
	float* d_ov;
	cudaMalloc(&d_ov, na * sizeof(float));
	cudaMemcpy(d_ov, ov, na * sizeof(float), cudaMemcpyHostToDevice);
	float* d_oa;
	cudaMalloc(&d_oa, na * sizeof(float));
	cudaMemcpy(d_oa, oa, na * sizeof(float), cudaMemcpyHostToDevice);
	
	kernel_cone <<< dim3(nx,ny,1), dim3(nz,1,1) >>> (d_reconstruction_array, tex_sinogram_array, d_transformation, na, su, sv, du, dv, d_ou, d_ov, d_oa, s2d);
	cudaMemcpy(reconstruction_array, d_reconstruction_array, nx*ny*nz*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_reconstruction_array);
	cudaFree(d_transformation);
	cudaFree(d_ou);
	cudaFree(d_ov);
	cudaFree(d_oa);
	cudaFreeArray(d_sinogram_array);
	cudaDestroyTextureObject(tex_sinogram_array);
}