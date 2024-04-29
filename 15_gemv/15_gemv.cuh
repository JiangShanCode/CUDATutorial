#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <string>
#include <stdexcept>

static const char* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);
}
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

template <typename T>
bool CheckResult(T *out, float *groudtruth, int M)
{
    for (int i = 0; i < M; i++){
        if ((float)out[i] != groudtruth[i]){
            printf("%d th comparsion: %f and %f \n", i, (float)out[i], groudtruth[i]);
            return false;
        }
    }
    return true;
}

template<typename T>
struct Vec {
    static constexpr int size = 4;
};
template<>
struct Vec<half> {
    static constexpr int size = 8;
};

template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<>
struct SumOp<half> {
  __device__ __forceinline__ half operator()(const half& a, const half& b) const { return __hadd(a, b); }
};

template<template<typename> class ReductionOp, typename T>
__device__ __forceinline__ T warpReduce(T val){
    for(int mask = 16; mask > 0; mask >>= 1){
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}
// 把block reduce拆分为多个warp reduce来计算
template<template<typename> class ReductionOp, typename T>
__device__ __forceinline__ T blockReduce(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    // 向上进1，以防分配的线程数量小于32导致warp nums为0
    int warp_nums = (blockDim.x + 31) / 32;
    // 2048 / 32 = 64 (一个block最多launch 2048个线程，一共64个warp)
    static __shared__ float warpres[64]; 
    // block内每个warp reduce的结果，该结果保存在每个warp内的0号线程，所以L80用0号线程写入warp res
    val = warpReduce<ReductionOp, T>(val);
    if (lane_id == 0){
        warpres[warp_id] = val;
    }
    __syncthreads();
    // 最后把每个warp的结果再作一个reduce得到最终一个block的结果
    float warp_val = tid < warp_nums ? warpres[tid] : 0;
    return warpReduce<ReductionOp, T>(warp_val);
}



