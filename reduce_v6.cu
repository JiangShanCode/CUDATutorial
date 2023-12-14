#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

// multi-block reduce two pass
// latency: 1.815ms

template <int blockSize>
__device__ void BlockSharedMemReduce(int* smem) {
  if (blockSize >= 1024) {
    if (threadIdx.x < 512) {
      smem[threadIdx.x] += smem[threadIdx.x + 512];
    }
    __syncthreads();
  }
  if (blockSize >= 512) {
    if (threadIdx.x < 256) {
      smem[threadIdx.x] += smem[threadIdx.x + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (threadIdx.x < 128) {
      smem[threadIdx.x] += smem[threadIdx.x + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (threadIdx.x < 64) {
      smem[threadIdx.x] += smem[threadIdx.x + 64];
    }
    __syncthreads();
  }
  // the final warp
  if (threadIdx.x < 32) {
    volatile int* vshm = smem;
    if (blockDim.x >= 64) {
      vshm[threadIdx.x] += vshm[threadIdx.x + 32];
    }
    vshm[threadIdx.x] += vshm[threadIdx.x + 16];
    vshm[threadIdx.x] += vshm[threadIdx.x + 8];
    vshm[threadIdx.x] += vshm[threadIdx.x + 4];
    vshm[threadIdx.x] += vshm[threadIdx.x + 2];                                                                                                                                                                                          vshm[threadIdx.x] += vshm[threadIdx.x + 1];

  }
}

template <int blockSize>
__global__ void reduce_v6(int *d_in, int *d_out, int nums){
    __shared__ int smem[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_thread_num = blockDim.x * gridDim.x;
    // unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // smem[tid] = d_in[i] + d_in[i + blockDim.x];
    // load: 每个线程负责若干个元素的thread local求和，最后存到shared mem对应位置
    int sum = 0;
    for (int32_t i = gtid; i < nums; i += total_thread_num) {
        sum += d_in[i];
    }
    smem[tid] = sum;
    __syncthreads();
    // compute: reduce in shared mem
    BlockSharedMemReduce<blockSize>(smem);

    // store: write back to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}

bool CheckResult(int *out, int groudtruth, int n){
    if (*out != groudtruth) {
      return false;
    }
    return true;
}

int main(){
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxblocks = deviceProp.maxGridSize[0];
    const int blockSize = 256;
    const int N = 25600000;

    // int gridSize = std::min((N + blockSize - 1) / blockSize, maxblocks);
    int gridSize = 100000 / 2;

    float milliseconds = 0;
    int *a = (int *)malloc(N * sizeof(int));
    int *d_a;
    cudaMalloc((void **)&d_a,N * sizeof(int));

    int *out = (int*)malloc((gridSize) * sizeof(int));
    int *d_out;
    int *part_out;//新增part_out存储每个block reduce的结果
    cudaMalloc((void **)&d_out, 1 * sizeof(int));
    cudaMalloc((void **)&part_out, (gridSize) * sizeof(int));
    int groudtruth = 0;

    for(int i = 0; i < N; i++){
        a[i] = rand() % 32;
        groudtruth += a[i];
    }
    printf("groudtruth %d \n",groudtruth);

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 Grid(gridSize);
    dim3 Block(blockSize);
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v6<blockSize><<<Grid, Block>>>(d_a, part_out, N);
    reduce_v6<blockSize><<<1, Block>>>(part_out, d_out, gridSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(out, groudtruth, 1);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0;i < 1;i++){
            printf("%d ",out[i]);
        }
        printf("\n");
    }
    printf("reduce_v6 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(part_out);
    free(a);
    free(out);
}
