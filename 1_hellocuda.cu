#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
// nvcc -arch sm_86 

// 被CPU启动,在GPU上执行
__global__ void hello_cuda(){
    // blockDim block内线程数
    // blockIdx blick的ID
    // threadIdx block内线程id
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("block id = [ %d ], thread id = [ %d ] hello cuda\n", blockIdx.x, idx);
}

int main() {
    // 2(block) * 10(每个block内线程) 个线程
    hello_cuda<<< 2, 1 >>>();
    printf("Hello World From CPU!\n");
    // 强制同步
    cudaDeviceSynchronize();
    return 0;
}

// int main(void){
//     // hello from cpu
//     printf("Hello World From CPU!\n");
//     return 0;
// }