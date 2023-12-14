#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"


//latency: 0.694ms
__device__ void WarpSharedMemReduce(volatile int* smem, int tid){
    // __syncwarp () 目的在于将共享内存读写分开
    int x = smem[tid];
    if (blockDim.x >= 64) {
      x += smem[tid + 32]; __syncwarp();
      smem[tid] = x; __syncwarp();
    }
    x += smem[tid + 16]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 8]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 4]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 2]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 1]; __syncwarp();
    smem[tid] = x; __syncwarp();
}
// Note: using blockSize as a template arg can benefit from NVCC compiler optimization, 
// which is better than using blockDim.x that is known in runtime.
template<int blockSize>
__global__ void reduce_v4(int *d_in,int *d_out){
    __shared__ int smem[blockSize];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    // load: 每个线程加载两个元素到shared mem对应位置
    smem[tid] = d_in[i] + d_in[i + blockSize];
    // 同步一个block 所有thread
    __syncthreads();

    // compute: reduce in shared mem
    // 思考这里是如何并行的
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        WarpSharedMemReduce(smem, tid);
    }

    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}

bool CheckResult(int *out, int groudtruth, int n){
    int res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }
    //printf("%f", res);
    if (res != groudtruth) {
        return false;
    }
    return true;
}

int main(){
    float milliseconds = 0;
    //const int N = 32 * 1024 * 1024;
    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    //int GridSize = 100000;
    int *a = (int *)malloc(N * sizeof(int));
    int *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(int));

    int *out = (int*)malloc((GridSize) * sizeof(int));
    int *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(int));

    int groudtruth = .0f;
    for(int i = 0; i < N; i++){
        a[i] = rand() % 32;
        groudtruth += a[i];
    }

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize / 2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v4<blockSize / 2><<<Grid,Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(int), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d \n", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        // for(int i = 0; i < GridSize;i++){
        //     printf("resPerBlock : %lf ",out[i]);
        // }
        // printf("\n");
        printf("groudtruth is: %d \n", groudtruth);
    }
    printf("reduce_v4 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
