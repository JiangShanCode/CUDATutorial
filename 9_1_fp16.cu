#include "cuda_fp16.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <iostream>
// #include <device_launch_parameters.h>

#define CHECK(call) \
{ \
	const cudaError_t error = call; \
	if (error != cudaSuccess) \
	{ \
		printf("Error: %s: %d, ", __FILE__, __LINE__); \
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
	} \
}
//size为转换前float数据个数，转换后由size/2个half2存储所有数据
__global__ void float22Half2Vec(float2 *src, __half2 *des, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = gridDim.x*blockDim.x;
	for (int i = idx; i<size / 2; i += stride)
		des[i] = __float22half2_rn(src[i]); 
}
__global__ void half22Float2Vec(__half2 *src, float2 *des, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = gridDim.x*blockDim.x;
	for (int i = idx; i<size / 2; i += stride)
		des[i] = __half22float2(src[i]);
}
//size为数据的多少，一共有size/2个half2型数据
__global__ void Half2Add(__half2 *a, __half2 *b, __half2 *c, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = gridDim.x*blockDim.x;
	for (int i = idx; i<size / 2; i += stride)
		c[i] = __hadd2(a[i], b[i]);
}


__global__ void HalfAdd(__half2 *a, __half2 *b, __half2 *c, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = gridDim.x*blockDim.x;
	for (int i = idx; i<size / 2; i += stride){
		c[i].x = a[i].x + b[i].x;
		c[i].y = a[i].y + b[i].y;
	}
}

__global__ void FP32Add(float *a,float *b,float *c,int size){
	int gtid = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = gridDim.x*blockDim.x;
	for (int i = gtid; i<size; i += stride)
		c[i] = a[i] + b[i];
}

int main()
{
	const int blocks = 128;
	const int threads = 128;
	size_t size = blocks * threads * 16;
	float * vec1 = new float[size];
	float * vec2 = new float[size];
	float * res = new float[size];
	float *h_resFP32 = new float[size];
	for (int i = 0; i < size; i++)
	{
		vec2[i] = vec1[i] = i;
	}
	float * vecDev1, *vecDev2, *resDev;
	CHECK(cudaMalloc((void **)&vecDev1, size * sizeof(float)));
	CHECK(cudaMalloc((void **)&vecDev2, size * sizeof(float)));
	CHECK(cudaMalloc((void **)&resDev, size * sizeof(float)));
	
	CHECK(cudaMemcpy(vecDev1, vec1, size * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(vecDev2, vec2, size * sizeof(float), cudaMemcpyHostToDevice));
 
 
	__half2 *vecHalf2Dev1, *vecHalf2Dev2, *resHalf2Dev;
	float *d_resFP32;

	CHECK(cudaMalloc((void **)&vecHalf2Dev1, size * sizeof(float) / 2));
	CHECK(cudaMalloc((void **)&vecHalf2Dev2, size * sizeof(float) / 2));
	CHECK(cudaMalloc((void **)&resHalf2Dev, size * sizeof(float) / 2));
	CHECK(cudaMalloc((void **)&d_resFP32, size * sizeof(float)));

	FP32Add<<<128,128>>>(vecDev1,vecDev2,d_resFP32,size);
	CHECK(cudaMemcpy(h_resFP32, d_resFP32, size * sizeof(float), cudaMemcpyDeviceToHost))

	float22Half2Vec << <128, 128 >> >((float2*)vecDev1, vecHalf2Dev1, size);
	float22Half2Vec << <128, 128 >> >((float2*)vecDev2, vecHalf2Dev2, size);

	Half2Add << <128, 128 >> > (vecHalf2Dev1, vecHalf2Dev2, resHalf2Dev, size);
	HalfAdd << <128, 128 >> > (vecHalf2Dev1, vecHalf2Dev2, resHalf2Dev, size);

	half22Float2Vec << <128, 128 >> >(resHalf2Dev, (float2*)resDev, size);
	CHECK(cudaMemcpy(res, resDev, size * sizeof(float), cudaMemcpyDeviceToHost));
	

	for (int i = 0; i < 128 * 128 ; i++)//打印出前64个结果，并与CPU结果对比
	{
        // half [-65504,65504]
		std::cout << vec1[i] << " + " << vec2[i] << " = " << h_resFP32[i] << "  ?  " << res[i] << std::endl;
	}
	delete[] vec1;
	delete[] vec2;
	delete[] res;
	delete[] h_resFP32;
	CHECK(cudaFree(vecDev1));
	CHECK(cudaFree(vecDev2));
	CHECK(cudaFree(resDev));
	CHECK(cudaFree(vecHalf2Dev1));
	CHECK(cudaFree(vecHalf2Dev2));
	CHECK(cudaFree(resHalf2Dev));
	CHECK(cudaFree(d_resFP32))
	// system("pause");
	return 0;
}