#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

//#define THREAD_PER_BLOCK 256
//屏蔽掉blockSize带来的forloop的编译优化为3.065ms
// bank conflict
template<int blockSize>
__global__ void reduce_v1(int *d_in,int *d_out){
    __shared__ int smem[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * blockSize + threadIdx.x;
    // load: 每个线程加载一个元素到shared mem对应位置
    smem[tid] = d_in[gtid];
    __syncthreads();

    // compute: reduce in shared mem
    // 思考这里是如何并行的
    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            smem[index] += smem[index + s];
        }
        __syncthreads();
    }

    // store: write back to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}

bool CheckResult(int *out, int groudtruth, int n){
    int res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }
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

    printf("groudtruth %d\n",groudtruth);
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v1<blockSize><<<Grid,Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(int), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d\n", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        //for(int i = 0; i < GridSize;i++){
            //printf("res per block : %lf ",out[i]);
        //}
        //printf("\n");
        printf("groudtruth is: %d \n", groudtruth);
    }
    printf("reduce_v1 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}

