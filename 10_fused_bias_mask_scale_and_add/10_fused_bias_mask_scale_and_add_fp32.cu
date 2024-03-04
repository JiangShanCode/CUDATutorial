#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <algorithm>

// TODO : what?
template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];
  __host__ __device__ inline const T& operator[](int i) const { return val[i]; }
  __host__ __device__ inline T& operator[](int i) { return val[i]; }
};

template<typename T>
struct MaskScaleAndElementwiseAddFunctor {
  MaskScaleAndElementwiseAddFunctor(const uint8_t* mask, const T* add_val, float scale)
      : mask(mask), add_val(add_val), scale(scale) {}
  __host__ __device__ T Compute(T x, int64_t i) const{
    return x * static_cast<T>(static_cast<bool>(mask[i]) * scale) + add_val[i];
  }
  const uint8_t* mask;
  const T* add_val;
  float scale;
};

// 0.990366ms
template<typename FUNCTOR, typename T>
__global__ void FusedBiasAddCUDAKernelFloat(FUNCTOR functor, const int elem_cnt, const int bias_size,
                                const T* x, const T* bias, T* y) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elem_cnt;
       i += blockDim.x * gridDim.x){
    T x_i = x[i] + bias[i % bias_size];
    // printf("%f\n",x_i);
    y[i] = functor.Compute(x_i, i);
  }
}

// Vec 没快多少? 0.989085ms
template<typename FUNCTOR, typename T,int Vecsize>
__global__ void FusedBiasAddCUDAKernelFloatVec(FUNCTOR functor, const int elem_cnt, const int bias_size,
                               const T* x, const T* bias, T* y) {

  int stride = blockDim.x * gridDim.x * 4;
  int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  int offset = gtid * 4;
  float y_reg[Vecsize];
  // 通过下标是否越界判断条件
  for (; offset < elem_cnt;offset += stride){
    using ArrT = AlignedVector<float, Vecsize>;

    // 向量化的读写
    const ArrT* in_arr = reinterpret_cast<const ArrT*>(x + offset);
    const float* in = reinterpret_cast<const float*>(in_arr);

    for (int i=0;i<Vecsize;i++){
      y_reg[i] = functor.Compute(in[i] + bias[(offset+i)%bias_size],offset+i);
    }
    *reinterpret_cast<ArrT*>(y+offset) = *reinterpret_cast<ArrT*>(y_reg);
  } 
}

template<typename FUNCTOR>
__global__ void myFusedBiasAddCUDAKernelFloatVec(FUNCTOR functor, const int elem_cnt, const int bias_size,
                               float* x, const float* bias, float* y) {

  int stride = blockDim.x * gridDim.x * 4;
  int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  int offset = gtid * 4;

  // 通过下标是否越界判断条件
  for (; offset < elem_cnt;offset += stride){
    float4 x4 = reinterpret_cast<float4 *>(x)[offset / 4];
    float4 y4;

    y4.x = functor.Compute(x4.x + bias[(offset)%bias_size],offset);
    y4.y = functor.Compute(x4.y + bias[(offset+1)%bias_size],offset+1);
    y4.z = functor.Compute(x4.z + bias[(offset+2)%bias_size],offset+2);
    y4.w = functor.Compute(x4.w + bias[(offset+3)%bias_size],offset+3);

    reinterpret_cast<float4 *>(y)[offset / 4] = y4;
  } 
}


bool check(float *out_gpu,float *out_cpu,int size){
  for (int i=0; i < size;i++){
    if (out_cpu[i] != out_gpu[i]){
      printf("%d %f %f\n",i,out_cpu[i],out_gpu[i]);
      return false;
    }
  }
  return true;
}

int main(){
    const int ele_cnt = 25600000;
    const float scale = 0.5;
    // const int Vecsize = 1;

    uint8_t* mask_tensor = new uint8_t[ele_cnt];
    
    float* add_val = new float[ele_cnt];
    for (int i = 0; i < ele_cnt; i++){
        mask_tensor[i] = (uint8_t)(i);
        add_val[i] = (float)(i);
    }
    int bias_size = 10;
    float *x = (float*) malloc(sizeof(float) * ele_cnt);
    float *y = (float*) malloc(sizeof(float) * ele_cnt);

    float *y_cpu = (float*) malloc(sizeof(float) * ele_cnt);

    float *bias = (float*) malloc(sizeof(float) * bias_size);

    for (int i = 0; i < ele_cnt; i++){
      x[i] = (float)(i);
    }

    for (int i = 0;i <bias_size;i++){
      bias[i] = (float)i;
    }

    float *d_x, *d_y, *d_bias ,*d_add_val;
    uint8_t *d_mask_tensor;
    cudaMalloc((void **)&d_x, ele_cnt * sizeof(float));
    cudaMalloc((void **)&d_y, ele_cnt * sizeof(float));
    cudaMalloc((void **)&d_bias, bias_size * sizeof(float));
    cudaMalloc((void **)&d_mask_tensor, ele_cnt * sizeof(uint8_t));
    cudaMalloc((void **)&d_add_val, ele_cnt * sizeof(float));

    cudaMemcpy(d_x, x, sizeof(float) * ele_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float) * ele_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, sizeof(float) * bias_size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_mask_tensor, mask_tensor, sizeof(uint8_t) * ele_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_add_val, add_val, sizeof(float) * ele_cnt, cudaMemcpyHostToDevice);


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxblocks = deviceProp.maxGridSize[0];
    int blockSize = 256;
    // int gridSize = std::min<int>((ele_cnt + blockSize - 1) / blockSize, maxblocks);
    // int gridSize = std::min<int>((ele_cnt / 4 + blockSize - 1) / blockSize, maxblocks);
    int gridSize = 10000;

    MaskScaleAndElementwiseAddFunctor<float> mask_scale_and_elementwise_add_func_cpu(mask_tensor, add_val, scale);
    MaskScaleAndElementwiseAddFunctor<float> mask_scale_and_elementwise_add_func_gpu(d_mask_tensor, d_add_val, scale);

    // FusedBiasAddCUDAKernelFloat<MaskScaleAndElementwiseAddFunctor<float>,float> <<<gridSize , blockSize>>>(mask_scale_and_elementwise_add_func_gpu, ele_cnt, bias_size, d_x, d_bias, d_y);

    // FusedBiasAddCUDAKernelFloatVec<MaskScaleAndElementwiseAddFunctor<float>,float,1> <<<gridSize , blockSize>>>(mask_scale_and_elementwise_add_func_gpu, ele_cnt, bias_size, d_x, d_bias, d_y);

    // FusedBiasAddCUDAKernelFloatVec<MaskScaleAndElementwiseAddFunctor<float>,float,4> <<<gridSize , blockSize>>>(mask_scale_and_elementwise_add_func_gpu, ele_cnt, bias_size, d_x, d_bias, d_y);

    myFusedBiasAddCUDAKernelFloatVec<MaskScaleAndElementwiseAddFunctor<float>> <<<gridSize , blockSize>>>(mask_scale_and_elementwise_add_func_gpu, ele_cnt, bias_size, d_x, d_bias, d_y);
    
    cudaMemcpy(y, d_y, sizeof(float) * ele_cnt, cudaMemcpyDeviceToHost);

    for (int i = 0; i < ele_cnt; i++){
      y_cpu[i] = mask_scale_and_elementwise_add_func_cpu.Compute(x[i]+bias[i%bias_size],i);
      // printf("(%f + %f) * %d * %f + %f = %f ? %f\n",x[i],bias[i%bias_size],static_cast<bool>(mask_tensor[i]),scale,add_val[i],y_cpu[i],y[i]);
    }

    bool is_right = check(y,y_cpu,ele_cnt);
    if (is_right){
      printf("the ans is right\n");
    } else{
      printf("the ans is wrong\n");
    }

    free(x);
    free(y);
    free(bias);
    delete add_val;
    add_val = nullptr;
    delete mask_tensor;
    add_val = nullptr;
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_bias);
    cudaFree(d_add_val);
    cudaFree(d_mask_tensor);
}
