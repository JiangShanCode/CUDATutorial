#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <time.h> 
#include <cuda_profiler_api.h>
//latency: 1.632064 ms
template<int blockSize>
__global__ void reduce_v0(int *d_in,int *d_out){
    __shared__ int smem[blockSize];

    int tid = threadIdx.x;
    // printf("%d\n",tid);
    int gtid = blockIdx.x * blockSize + threadIdx.x;
    // load: 每个线程加载一个元素到shared mem对应位置
    smem[tid] = d_in[gtid];
    __syncthreads();

    // compute: reduce in shared mem
    // 思考这里是如何并行的
    for(int index = 1; index < blockDim.x; index *= 2) {
        if (tid % (2 * index) == 0) {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    // store: write back to global mem
    // 最后结果存在smem[0]上 
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}

// TODO: 在globel 上读数据
// 1.759072 ms
__global__ void reduce_v0_globelmem(int *d_in, int *d_out){
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int index = 1;index < blockDim.x;index <<= 1){
        if (gtid % (2 * index) == 0){
            d_in[gtid] += d_in[gtid+index];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        d_out[blockIdx.x] = d_in[blockIdx.x * blockDim.x];
    }
}

bool CheckResult(int *out, int groudtruth, int n){
    int res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
        // printf("%lld\n",out[i]);
    }
    if (res != groudtruth) {
        return false;
    }
    return true;
}

int main(){
    float milliseconds = 0;
    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    // TODO:如果取deviceProp.maxGridSize[0] 数据不就没处理完嘛?
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    //int GridSize = 100000;
    int *a = (int *)malloc(N * sizeof(int));
    // d_a 指向globel mem
    int *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(int));

    int *out = (int*)malloc((GridSize) * sizeof(int));
    int *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(int));

    int groudtruth = 0;
    // srand((unsigned)time(NULL)); 
    for(int i = 0; i < N; i++){
        // a[i] = i;
        a[i] = rand() % 32;
        groudtruth += a[i];
    }
    printf("groudtruth %d\n",groudtruth);

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);
    cudaProfilerStart();
    reduce_v0<blockSize><<<Grid,Block>>>(d_a, d_out);
    // reduce_v0_globelmem<<<Grid,Block>>>(d_a,d_out);
    cudaProfilerStop();

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("reduce_v0 latency = %f ms\n", milliseconds);

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
    

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
