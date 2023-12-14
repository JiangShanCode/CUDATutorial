#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#define BLOCKSIZE 256

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/********************/
/* ADD_FLOAT KERNEL */
/********************/
__global__ void add_float(float *d_a, float *d_b, float *d_c, unsigned int N) {

    const int tid = 4 * threadIdx.x + blockIdx.x * (4 * blockDim.x);

    if (tid < N) {

        float a1 = d_a[tid];
        float b1 = d_b[tid];

        float a2 = d_a[tid+1];
        float b2 = d_b[tid+1];

        float a3 = d_a[tid+2];
        float b3 = d_b[tid+2];

        float a4 = d_a[tid+3];
        float b4 = d_b[tid+3];

        float c1 = a1 + b1;
        float c2 = a2 + b2;
        float c3 = a3 + b3;
        float c4 = a4 + b4;

        d_c[tid] = c1;
        d_c[tid+1] = c2;
        d_c[tid+2] = c3;
        d_c[tid+3] = c4;

        //if ((tid < 1800) && (tid > 1790)) {
            //printf("%i %i %i %f %f %f\n", tid, threadIdx.x, blockIdx.x, a1, b1, c1);
            //printf("%i %i %i %f %f %f\n", tid+1, threadIdx.x, blockIdx.x, a2, b2, c2);
            //printf("%i %i %i %f %f %f\n", tid+2, threadIdx.x, blockIdx.x, a3, b3, c3);
            //printf("%i %i %i %f %f %f\n", tid+3, threadIdx.x, blockIdx.x, a4, b4, c4);
        //}

    }

}

/*********************/
/* ADD_FLOAT2 KERNEL */
/*********************/
__global__ void add_float2(float2 *d_a, float2 *d_b, float2 *d_c, unsigned int N) {

    const int tid = 2 * threadIdx.x + blockIdx.x * (2 * blockDim.x);

    if (tid < N) {

        float2 a1 = d_a[tid];
        float2 b1 = d_b[tid];

        float2 a2 = d_a[tid+1];
        float2 b2 = d_b[tid+1];

        float2 c1;
        c1.x = a1.x + b1.x;
        c1.y = a1.y + b1.y;

        float2 c2;
        c2.x = a2.x + b2.x;
        c2.y = a2.y + b2.y;

        d_c[tid] = c1;
        d_c[tid+1] = c2;

    }

}

/*********************/
/* ADD_FLOAT4 KERNEL */
/*********************/
__global__ void add_float4(float4 *d_a, float4 *d_b, float4 *d_c, unsigned int N) {

    const int tid = 1 * threadIdx.x + blockIdx.x * (1 * blockDim.x);

    if (tid < N/4) {

        float4 a1 = d_a[tid];
        float4 b1 = d_b[tid];

        float4 c1;
        c1.x = a1.x + b1.x;
        c1.y = a1.y + b1.y;
        c1.z = a1.z + b1.z;
        c1.w = a1.w + b1.w;

        d_c[tid] = c1;

    }

}

/********/
/* MAIN */
/********/
int main() {

    const int N = 8*100000000; //10M

    const float a = 3.f;
    const float b = 5.f;
    float *res = new float[N];
    for (int i = 0;i < N;i++){
        res[i] = 8.f;
    }

    // --- float

    thrust::device_vector<float> d_A(N, a);
    thrust::device_vector<float> d_B(N, b);
    thrust::device_vector<float> d_C(N);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    add_float<<<iDivUp(N/4, BLOCKSIZE), BLOCKSIZE>>>(thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()), thrust::raw_pointer_cast(d_C.data()), N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("float Elapsed time:  %3.1f ms \n", time); gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    thrust::host_vector<float> h_float = d_C;
    for (int i=0; i<N; i++) {
        if (h_float[i] != res[i]) {
            printf("Error for add_float at %i: result is %f\n",i, h_float[i]);
            return -1;
        }
    }

    // --- float2
    
    // thrust::device_vector<float> d_A2(N, a);
    // thrust::device_vector<float> d_B2(N, b);
    // thrust::device_vector<float> d_C2(N);

    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    add_float2<<<iDivUp(N/4, BLOCKSIZE), BLOCKSIZE>>>((float2*)thrust::raw_pointer_cast(d_A.data()), (float2*)thrust::raw_pointer_cast(d_B.data()), (float2*)thrust::raw_pointer_cast(d_C.data()), N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("float2 Elapsed time:  %3.1f ms \n", time); gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    thrust::host_vector<float> h_float2 = d_C;
    for (int i=0; i<N; i++) {
        if (h_float2[i] != res[i]) {
            printf("Error for add_float2 at %i: result is %f\n",i, h_float2[i]);
            return -1;
        }
    }

    // --- float4

    // thrust::device_vector<float> d_A4(N, a);
    // thrust::device_vector<float> d_B4(N, b);
    // thrust::device_vector<float> d_C4(N);

    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    add_float4<<<iDivUp(N/4, BLOCKSIZE), BLOCKSIZE>>>((float4*)thrust::raw_pointer_cast(d_A.data()), (float4*)thrust::raw_pointer_cast(d_B.data()), (float4*)thrust::raw_pointer_cast(d_C.data()), N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("float4 Elapsed time:  %3.1f ms \n", time); gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    thrust::host_vector<float> h_float4 = d_C;
    for (int i=0; i<N; i++) {
        if (h_float4[i] != res[i]) {
            printf("Error for add_float4 at %i: result is %f\n",i, h_float4[i]);
            return -1;
        }
    }

    return 0;
}