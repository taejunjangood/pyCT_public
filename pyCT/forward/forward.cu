#include "forward.h"

__global__ 
void kernel_parallel(float* proj, cudaTextureObject_t texObjImg, float* transformation, int nw)
{
	int nu = gridDim.x;
	int nv = gridDim.y;
	int u = blockIdx.x;
	int v = blockIdx.y;
	int a = threadIdx.x;

	float t00 = transformation[0 + 0*4 + a*4*4];
	float t01 = transformation[1 + 0*4 + a*4*4];
	float t02 = transformation[2 + 0*4 + a*4*4];
	float t03 = transformation[3 + 0*4 + a*4*4];

	float t10 = transformation[0 + 1*4 + a*4*4];
	float t11 = transformation[1 + 1*4 + a*4*4];
	float t12 = transformation[2 + 1*4 + a*4*4];
	float t13 = transformation[3 + 1*4 + a*4*4];

	float t20 = transformation[0 + 2*4 + a*4*4];
	float t21 = transformation[1 + 2*4 + a*4*4];
	float t22 = transformation[2 + 2*4 + a*4*4];
	float t23 = transformation[3 + 2*4 + a*4*4];

	float xx = t00 * u + t01 * v + t03;
	float yy = t10 * u + t11 * v + t13;
	float zz = t20 * u + t21 * v + t23;

	float sum = 0;
	float x, y, z;

	for (int w = 0; w < nw; w++)
	{
		x = xx + t02 * w;
		y = yy + t12 * w;
		z = zz + t22 * w;
		sum += tex3D<float>(texObjImg, x+.5, y+.5, z+.5);
	}
	int idx = u + v*nu + a*nu*nv;
	proj[idx] = sum;
}

void funcParallelBeam(float* detector_array, float* object_array, float* transformation, int nx, int ny, int nz, int nu, int nv, int nw, int na)
{
	// object array >> texture memory
	const cudaExtent objSize = make_cudaExtent(nx, ny, nz);
	cudaArray* d_object_array = 0;
	cudaTextureObject_t tex_object_array = 0;

	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMalloc3DArray(&d_object_array, &channelDesc, objSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)object_array, objSize.width * sizeof(float), objSize.width, objSize.height);
	copyParams.dstArray = d_object_array;
	copyParams.extent = objSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_object_array;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = false; // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear; // linear interpolation
	texDescr.addressMode[0] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.addressMode[1] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.addressMode[2] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&tex_object_array, &texRes, &texDescr, NULL);

	float* d_transformation;
	cudaMalloc(&d_transformation, na * 4 * 4 * sizeof(float));
	cudaMemcpy(d_transformation, transformation, na * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
	float* d_detector_array;
	cudaMalloc(&d_detector_array, na * nu * nv * sizeof(float));

	kernel_parallel <<< dim3(nu,nv,1), dim3(na,1,1) >>> (d_detector_array, tex_object_array, d_transformation, nw);
	cudaMemcpy(detector_array, d_detector_array, na*nu*nv*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_detector_array);
	cudaFree(d_transformation);
	cudaFreeArray(d_object_array);
	cudaDestroyTextureObject(tex_object_array);
}


__global__ 
void kernel_cone(float* proj, cudaTextureObject_t texObjImg, float* transformation, int nw, float su, float sv, float* ou, float* ov, float* oa, float s2d, float near, float far)
{
	int nu = gridDim.x;
	int nv = gridDim.y;
	int u = blockIdx.x;
	int v = blockIdx.y;
	int a = threadIdx.x;

	float rx_ = -su/2 + su/2/nu + u*su/nu;
	float ry_ = -sv/2 + sv/2/nv + v*sv/nv;
	
	float rx = rx_*cosf(oa[a]) + ry_*sinf(oa[a]) + ou[a];
	float ry = -rx_*sinf(oa[a]) + ry_*cosf(oa[a]) + ov[a];
	float rz = -s2d;
	float magnitude = powf((powf(rx,2.) + powf(ry,2.) + powf(rz,2.)), .5);
	rx /= magnitude;
	ry /= magnitude;
	rz /= magnitude;

	float t = near;
	float dt = (far - near) / nw;
	if (rz*far > s2d)
	{
		nw = (int) (s2d/rz-near)/dt;
	}

	float t00 = transformation[0 + 0*4 + a*4*4];
	float t01 = transformation[1 + 0*4 + a*4*4];
	float t02 = transformation[2 + 0*4 + a*4*4];
	float t03 = transformation[3 + 0*4 + a*4*4];

	float t10 = transformation[0 + 1*4 + a*4*4];
	float t11 = transformation[1 + 1*4 + a*4*4];
	float t12 = transformation[2 + 1*4 + a*4*4];
	float t13 = transformation[3 + 1*4 + a*4*4];

	float t20 = transformation[0 + 2*4 + a*4*4];
	float t21 = transformation[1 + 2*4 + a*4*4];
	float t22 = transformation[2 + 2*4 + a*4*4];
	float t23 = transformation[3 + 2*4 + a*4*4];

	float sum = 0;
	float x, y, z;

	for (int i = 0; i <= nw; i++)
	{
		x = t00*rx*t + t01*ry*t + t02*rz*t + t03;
		y = t10*rx*t + t11*ry*t + t12*rz*t + t13;
		z = t20*rx*t + t21*ry*t + t22*rz*t + t23;
		sum += tex3D<float>(texObjImg, x+.5, y+.5, z+.5);
		t += dt;
	}
	int idx = u + v*nu + a*nu*nv;
	proj[idx] = sum;
}

void funcConeBeam(float* detector_array, float* object_array, float* transformation, int nx, int ny, int nz, int nu, int nv, int nw, int na, float su, float sv, float* ou, float* ov, float* oa, float s2d, float near, float far)
{
	// object array >> texture memory
	const cudaExtent objSize = make_cudaExtent(nx, ny, nz);
	cudaArray* d_object_array = 0;
	cudaTextureObject_t tex_object_array = 0;

	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMalloc3DArray(&d_object_array, &channelDesc, objSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)object_array, objSize.width * sizeof(float), objSize.width, objSize.height);
	copyParams.dstArray = d_object_array;
	copyParams.extent = objSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_object_array;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = false; // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear; // linear interpolation
	texDescr.addressMode[0] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.addressMode[1] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.addressMode[2] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&tex_object_array, &texRes, &texDescr, NULL);

	float* d_detector_array;
	cudaMalloc(&d_detector_array, na * nu * nv * sizeof(float));
	float* d_transformation;
	cudaMalloc(&d_transformation, na * 4 * 4 * sizeof(float));
	cudaMemcpy(d_transformation, transformation, na * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
	float* d_ou;
	cudaMalloc(&d_ou, na * sizeof(float));
	cudaMemcpy(d_ou, ou, na * sizeof(float), cudaMemcpyHostToDevice);
	float* d_ov;
	cudaMalloc(&d_ov, na * sizeof(float));
	cudaMemcpy(d_ov, ov, na * sizeof(float), cudaMemcpyHostToDevice);
	float* d_oa;
	cudaMalloc(&d_oa, na * sizeof(float));
	cudaMemcpy(d_oa, oa, na * sizeof(float), cudaMemcpyHostToDevice);

	kernel_cone <<< dim3(nu,nv,1), dim3(na,1,1) >>> (d_detector_array, tex_object_array, d_transformation, nw, su, sv, d_ou, d_ov, d_oa, s2d, near, far);
	cudaMemcpy(detector_array, d_detector_array, na*nu*nv*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_detector_array);
	cudaFree(d_transformation);
	cudaFree(d_ou);
	cudaFree(d_ov);
	cudaFree(d_oa);
	cudaFreeArray(d_object_array);
	cudaDestroyTextureObject(tex_object_array);
}