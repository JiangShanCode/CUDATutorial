#include <cuda_fp16.h>
// vec * mat, mat is row major
// [1, N] * [N, M]
// logits * v
// 有关fp32/fp16 fma和add的各种重载操作
namespace gemv2 {
    struct half8 {
        half2 h1;
        half2 h2;
        half2 h3;
        half2 h4;

        __device__ half8& operator = (half8 h8) {
            h1 = h8.h1;
            h2 = h8.h2;
            h3 = h8.h3;
            h4 = h8.h4;
            return *this;
        }
    };

    template<int M, typename T>
    struct get_threads_per_mat_row {
        static const int value = M * sizeof(T) / 16;
    };

    inline __device__ float add(float a, float b)
    {
        return a + b;
    }

    inline __device__ float4 add(float4 a, float4 b)
    {
        float4 c;
        c.x = gemv2::add(a.x, b.x);
        c.y = gemv2::add(a.y, b.y);
        c.z = gemv2::add(a.z, b.z);
        c.w = gemv2::add(a.w, b.w);
        return c;
    }
    inline __device__ half add(half a, half b)
    {
        return __hadd(a, b);
        //if use L216, half+half is not really adding, its so weird, which  cause our result is 32, not 256
        // return (half)((float)a+(float)b);
    }

    inline __device__ half2 add(half2 a, half2 b)
    {
        half2 res;
        res.x = gemv2::add(a.x, b.x);
        res.y = gemv2::add(a.y, b.y);
        return res;
    }

    inline __device__ half8 add(half8 a, half8 b)
    {
        half8 c;
        c.h1 = gemv2::add(a.h1, b.h1);
        c.h2 = gemv2::add(a.h2, b.h2);
        c.h3 = gemv2::add(a.h3, b.h3);
        c.h4 = gemv2::add(a.h4, b.h4);
        return c;
    }

    inline __device__ half fma(half a, half b, half c)
    {
        return __hadd(__hmul(a ,b), c);
        // 有时候编译器会不认识__hmul或者__hadd，所以粗暴转成fp32计算再转回fp16
        // return __float2half((float)a * (float)b + (float)c);
    }


    inline __device__ half2 fma(half a, half2 b, half2 c)
    {
        half2 res;
        res.x = gemv2::fma(a, b.x, c.x);
        res.y = gemv2::fma(a, b.y, c.y);
        return res;
    }

    inline __device__ half8 fma(half a, half8 b, half8 c)
    {
        half8 d;
        d.h1 = gemv2::fma(a, b.h1, c.h1);
        d.h2 = gemv2::fma(a, b.h2, c.h2);
        d.h3 = gemv2::fma(a, b.h3, c.h3);
        d.h4 = gemv2::fma(a, b.h4, c.h4);
        return d;
    }

    inline __device__ float fma(float a, float b, float c)
    {
        return a * b + c;
    }

    inline __device__ float4 fma(float a, float4 b, float4 c)
    {
        float4 d;
        d.x = gemv2::fma(a, b.x, c.x);
        d.y = gemv2::fma(a, b.y, c.y);
        d.z = gemv2::fma(a, b.z, c.z);
        d.w = gemv2::fma(a, b.w, c.w);
        return d;
    }
} // namespace gemv2

// 1个block处理一个[1, M], 循环处理完[N, M]
// for fp32: <64, M * sizeof(T) / 16 = M / 4, 4>
template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
__global__ void gemv2_kernel(float* matrix, float* vector, float* res, int N, int M) {
    //根据编译期常量获取每个thread处理的行列号
    int tid = threadIdx.x;
    // 每个线程负责数据所在行号
    int mat_o = tid / THREADS_PER_VALUE;
    // 每个线程负责数据所在向量号
    int mat_i = tid % THREADS_PER_VALUE * VEC_SIZE;
    // 一个block处理的行数
    constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
    __shared__ float out_smem[512];
    float4 out;
    // 点乘或fma，inter-block循环累加
    for (int ti = mat_o; ti < N; ti += ROW_PER_ITER) {
        float4 mat = *reinterpret_cast<float4*>(&matrix[ti * M + mat_i]);
        float logits = vector[ti];
        // fused mul and add: d = a * b + c
        out = gemv2::fma(logits, mat, out);
    }
    // intra-block二分法相加得最终结果
    for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
        int midpoint = ROWS_PER_BLOCK / 2;
        if (mat_o >= midpoint && mat_o < ROWS_PER_BLOCK) {
            *reinterpret_cast<float4*>(&out_smem[(mat_o - midpoint) * M + mat_i]) = out;
        }
        __syncthreads();
        if (mat_o < midpoint) {
            // ROW_PER_ITER中上半部分out和下半部分out相加
            out = gemv2::add(*reinterpret_cast<float4*>(&out_smem[mat_o * M + mat_i]), out);
        }
        __syncthreads();
    }
    // 二分法最终结果存在首行，写回显存
    if (mat_o == 0) {
        *reinterpret_cast<float4*>(&res[mat_i]) = out;
    }
}

// for fp16: <64, M * sizeof(T) / 16 = M / 8, 8>
template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
__global__ void gemv2_kernel(half* matrix, half* vector, half* res, int N, int M) {
    int tid = threadIdx.x;
    int mat_o = tid / THREADS_PER_VALUE;
    int mat_i = tid % THREADS_PER_VALUE * VEC_SIZE;
    constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
    __shared__ half out_smem[2048];
    gemv2::half8 out;
    // zero(out);
    for (int ti = mat_o; ti < N; ti += ROW_PER_ITER) {
        gemv2::half8 mat = *reinterpret_cast<gemv2::half8*>(&matrix[ti * M + mat_i]);
        half logits = vector[ti];
        out = gemv2::fma(logits, mat, out);
    }
    for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
        int midpoint = ROWS_PER_BLOCK / 2;
        if (mat_o >= midpoint && mat_o < ROWS_PER_BLOCK) {
            *reinterpret_cast<gemv2::half8*>(&out_smem[(mat_o - midpoint) * M + mat_i]) = out;
        }
        __syncthreads();

        if (mat_o < midpoint) {
            // ROW_PER_ITER中上半部分out和下半部分out相加
            out = gemv2::add(*reinterpret_cast<gemv2::half8*>(&out_smem[mat_o * M + mat_i]), out);
        }
        __syncthreads();
    }
    if (mat_o == 0) {
        *reinterpret_cast<gemv2::half8*>(&res[mat_i]) = out;
    }
}
// TODO: 修改float4部分为可以泛化表示float4和half8类型的代码, 而后此模板函数可以取代以上fp32和fp16的gemv2
template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE, typename T>
__global__ void gemv2_kernel_template(T* matrix, T* vector, T* res, int N, int M) {
    int tid = threadIdx.x;
    int mat_o = tid / THREADS_PER_VALUE;
    int mat_i = tid % THREADS_PER_VALUE * VEC_SIZE;
    constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
    __shared__ T out_smem[512];

    float4 out; //TODO
    for (int ti = mat_o; ti < N; ti += ROW_PER_ITER) {
        float4 mat = *reinterpret_cast<float4*>(&matrix[ti * M + mat_i]);//TODO
        T logits = vector[ti];
        out = gemv2::fma(logits, mat, out);
    }
    for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
        int midpoint = ROWS_PER_BLOCK / 2;
        if (mat_o >= midpoint && mat_o < ROWS_PER_BLOCK) {
            *reinterpret_cast<float4*>(&out_smem[(mat_o - midpoint) * M + mat_i]) = out;//TODO
        }
        __syncthreads();
        if (mat_o < midpoint) {
            // ROW_PER_ITER中上半部分out和下半部分out相加
            out = gemv2::add(*reinterpret_cast<float4*>(&out_smem[mat_o * M + mat_i]), out);//TODO
        }
        __syncthreads();
    }
    if (mat_o == 0) {
        *reinterpret_cast<float4*>(&res[mat_i]) = out;//TODO
    }
}

template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
struct DispatchLauncher2
{
    template<typename T>
    static void launcher(T* d_mat, T* d_vec, T* d_dst, int M, int N){
        dim3 Grid(1);
        dim3 Block(THREADS_PER_BLOCK);
        float milliseconds = 0;
        // 使用cudaevent计时，开销最小
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        printf("calling\n");
        // 启动cuda kernel
        gemv2_kernel<THREADS_PER_BLOCK, THREADS_PER_VALUE, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N, M);
        cudaError_t result = cudaGetLastError();
        if (result) {
            throw std::runtime_error(std::string("[ERROR] CUDA runtime error: ") +  (_cudaGetErrorEnum(result)) + " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");
        }
        printf("called\n");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("gemv latency = %f ms\n", milliseconds);
    }
};
