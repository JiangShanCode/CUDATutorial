#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sum(float *x)
{
    int block_id = blockIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    printf("current block=%d, thread id in current block =%d, global thread id=%d\n", block_id, local_tid, global_tid);
    x[global_tid] += 1;
}

int main(){
    int N = 32;
    int nbytes = N * sizeof(float);
    // *dx 指向显存, *hx 内存
    float *dx, *hx;
    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);//思考为什么要用二级指针
    /* allocate CPU mem */
    hx = (float*) malloc(nbytes);
    /* init host data */
    printf("hx original: \n");
    for (int i = 0; i < N; i++) {
        hx[i] = i;
        printf("%g\n", hx[i]);
    }
    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    /* launch GPU kernel */
    sum<<<1, N>>>(dx);
    /* copy data from GPU */
    cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);
    printf("hx current: \n");
    for (int i = 0; i < N; i++) {
        printf("%g\n", hx[i]);
    }
    cudaFree(dx);
    free(hx);
    return 0;
}
