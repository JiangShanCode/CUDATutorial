// 一个blk计算一个元素
// mat * vec = {M, N} * {N, 1}/{1, N}
template<int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv(float* matrix, float* vector, float* res, int cols) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float thread_local_sum = 0.0f;
    // printf("VECS_PER_THREAD %d\n",VECS_PER_THREAD);

    // 1. array of float4
    float4* mat4 = reinterpret_cast<float4*>(&matrix[bid * cols + tid * VEC_SIZE * VECS_PER_THREAD]); // 4 * half2
    float4* vec4 = reinterpret_cast<float4*>(&vector[tid * VEC_SIZE * VECS_PER_THREAD]);
    for(int i = 0; i < VECS_PER_THREAD; i++) {
        thread_local_sum += mat4[i].x * vec4[i].x;
        thread_local_sum += mat4[i].y * vec4[i].y;
        thread_local_sum += mat4[i].z * vec4[i].z;
        thread_local_sum += mat4[i].w * vec4[i].w;
    }

    // 2. float4
    // for(int i = 0; i < VECS_PER_THREAD; i++) {
    //     float4 mat4 = reinterpret_cast<float4*>(matrix)[bid * (cols / VEC_SIZE) + tid + i * blockDim.x]; // 4 * half2
    //     float4 vec4 = reinterpret_cast<float4*>(vector)[tid + i * blockDim.x];
    //     thread_local_sum += mat4.x * vec4.x;
    //     thread_local_sum += mat4.y * vec4.y;
    //     thread_local_sum += mat4.z * vec4.z;
    //     thread_local_sum += mat4.w * vec4.w;
    // }

    //reduce to get the final val
    float reduce_res = blockReduce<SumOp, float>(thread_local_sum);
    //store to gmem
    if(tid == 0) {
        res[blockIdx.x] = reduce_res;
    }
    __syncthreads();

}

template<int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv(half* matrix, half* vector, half* res, int cols) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    //float thread_local_sum = 0.0f;
    half thread_local_sum = 0;
    for(int i = 0; i < VECS_PER_THREAD; i++) {
        float4* mat4 = reinterpret_cast<float4*>(&matrix[bid * cols + i * VEC_SIZE * blockDim.x + tid * VEC_SIZE]); // 4 * half2
        float4* vec4 = reinterpret_cast<float4*>(&vector[i * VEC_SIZE * blockDim.x + tid * VEC_SIZE]);
        half2* vec_h1 = (half2*)&vec4[i].x;
        half2* vec_h2 = (half2*)&vec4[i].y;
        half2* vec_h3 = (half2*)&vec4[i].z;
        half2* vec_h4 = (half2*)&vec4[i].w;
        half2* mat_h1 = (half2*)&mat4[i].x;
        half2* mat_h2 = (half2*)&mat4[i].y;
        half2* mat_h3 = (half2*)&mat4[i].z;
        half2* mat_h4 = (half2*)&mat4[i].w;   
        half2 res1 = __hmul2(*mat_h1, *vec_h1);
        half2 res2 = __hmul2(*mat_h2, *vec_h2);
        half2 res3 = __hmul2(*mat_h3, *vec_h3);
        half2 res4 = __hmul2(*mat_h4, *vec_h4); 
        half2 res = __hadd2(__hadd2(__hadd2(res1, res2), res3), res4);
        thread_local_sum = __hadd(res.x, res.y);
        // float2 res1 = __half22float2(__hmul2(*mat_h1, *vec_h1));
        // float2 res2 = __half22float2(__hmul2(*mat_h2, *vec_h2));
        // float2 res3 = __half22float2(__hmul2(*mat_h3, *vec_h3));
        // float2 res4 = __half22float2(__hmul2(*mat_h4, *vec_h4));
        // thread_local_sum += res1.x;
        // thread_local_sum += res1.y;
        // thread_local_sum += res2.x;
        // thread_local_sum += res2.y;
        // thread_local_sum += res3.x;
        // thread_local_sum += res3.y;
        // thread_local_sum += res4.x;
        // thread_local_sum += res4.y;
        if(i == 0 && tid == 0 && bid == 0) {
            printf("thread sum = %f\n", (float)thread_local_sum); // 8
            // printf("res1.x = %f\n", res1.x); // 1
            // printf("res1.y = %f\n", res1.y);
        }
    }
    //reduce to get the final val
    half reduce_res = blockReduce<SumOp, half>(thread_local_sum);
    // float reduce_res = blockReduce<SumOp, float>(thread_local_sum);
    //store to gmem
    if(tid == 0) {
        printf("block reduce_res = %f\n", (float)reduce_res);
        // res[blockIdx.x] = __float2half(reduce_res);
        res[blockIdx.x] = reduce_res;
    }
    __syncthreads();
}


template<int VECS_PER_THREAD, int VEC_SIZE, int THREAD_NUMS>
struct DispatchLauncher
{
    template<typename T>
    static void launcher(T* d_mat, T* d_vec, T* d_dst, int M, int N){
        dim3 Grid(M);
        // dim3 Grid(1);
        dim3 Block(THREAD_NUMS);
        float milliseconds = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        printf("calling\n");
        gemv<VECS_PER_THREAD, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N);
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