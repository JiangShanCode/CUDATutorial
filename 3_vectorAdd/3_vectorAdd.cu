#include <stdio.h>
#include <cuda.h>
#include <algorithm>
#include <cuda_runtime.h>

#define ARRAY_SIZE MEMORY_OFFSET * BENCH_ITER //Array size has to exceed L2 size to avoid L2 cache residence
#define MEMORY_OFFSET 1000000 // 4M
#define BENCH_ITER 10
#define THREADS_NUM 256

/* CUDA kernel function */
__global__ void vec_add(float *x, float *y, float *z)
{
    /* 2D grid */
    // int idx = (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x);
    /* 1D grid */
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("%d\n",idx);
    for(int i = idx;i < MEMORY_OFFSET;i += blockDim.x * gridDim.x){       
        z[i] = x[i] + y[i];
    }
}

void vec_add_cpu(float *x, float *y, float *z)
{
    for (int i = 0; i < ARRAY_SIZE; i++) z[i] = y[i] + x[i];
}

int main()
{
    float *A = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *B = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *C = (float*) malloc(ARRAY_SIZE*sizeof(float));

	float *A_g;
	float *B_g;
	float *C_g;

	float milliseconds = 0;

	for (uint32_t i=0; i<ARRAY_SIZE; i++){
		A[i] = (float)i;
		B[i] = (float)i;
	}
	cudaMalloc((void**)&A_g, ARRAY_SIZE*sizeof(float));
	cudaMalloc((void**)&B_g, ARRAY_SIZE*sizeof(float));
	cudaMalloc((void**)&C_g, ARRAY_SIZE*sizeof(float));

	cudaMemcpy(A_g, A, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_g, B, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxblocks = deviceProp.maxGridSize[0];
	int BlockNums = std::min<int>((MEMORY_OFFSET + THREADS_NUM - 1) / THREADS_NUM,maxblocks);

    printf("warm up start\n");
	vec_add<<<BlockNums, THREADS_NUM>>>(A_g, B_g, C_g);
	printf("warm up end\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    /* launch GPU kernel */
    for (int i = BENCH_ITER - 1; i >= 0; --i) {
		vec_add<<<BlockNums, THREADS_NUM>>>(A_g, B_g, C_g);
	}
    cudaEventRecord(stop);
    // CPU 等待stop 记录
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);  
    
	/* copy GPU result to CPU */
    cudaMemcpy(C, C_g, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    /* CPU compute */
    float* C_cpu_res = (float *) malloc(ARRAY_SIZE*sizeof(float));
    vec_add_cpu(A, B, C_cpu_res);

    /* check GPU result with CPU*/
    for (int i = 0; i < MEMORY_OFFSET; ++i) {
        if (fabs(C_cpu_res[i] - C[i]) > 1e-6) {
            printf("%f %f\n",C_cpu_res[i],C[i]);
            printf("Result verification failed at element index %d!\n", i);
        }
    }
    printf("Result right\n");
    cudaFree(A_g);
    cudaFree(B_g);
    cudaFree(C_g);

    free(A);
    free(B);
    free(C);
    free(C_cpu_res);

    return 0;
}