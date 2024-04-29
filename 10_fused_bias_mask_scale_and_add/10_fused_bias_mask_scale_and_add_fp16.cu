#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include <algorithm>

template<typename T>
struct MaskScaleAndElementwiseAddFunctor {
  MaskScaleAndElementwiseAddFunctor(const uint8_t* mask, const T* add_val, float scale)
      : mask(mask), add_val(add_val), scale(scale) {}
  __host__ __device__ T Compute(T x, int64_t i) const {
    return x * static_cast<T>(static_cast<bool>(mask[i]) * scale) + add_val[i];
  }
  const uint8_t* mask;
  const T* add_val;
  float scale;
};

template<>
struct MaskScaleAndElementwiseAddFunctor<half> {
  MaskScaleAndElementwiseAddFunctor(const uint8_t* mask, const half* add_val, float scale)
      : mask(mask), add_val(add_val), scale(scale) {}
  // half标量版本的MaskScaleAndElementwiseAdd，与L15区别不大，注意: 有的GPU在有的nvcc和cuda版本下，没有重载half*half的直接相乘版本，此时需要用hmul(half,half)来代替或者两个half强转为fp32来相乘再转回half,比如(half)((float)x * (float)y)
  __device__ half Compute(half x, int64_t i) const {
    return x * static_cast<half>(static_cast<bool>(mask[i]) * scale) + add_val[i];
  }
  // half向量版本的MaskScaleAndElementwiseAdd，不仅支持L32和L33所示的向量化读取，也支持L39所示的向量化计算，这与fp32的向量化是不同的，具体接口可以搜索cuda math api文档
  __device__ half2 ComputeHalf2(half2 x, int64_t i) const {
    const char2* mask_c2 = reinterpret_cast<const char2*>(mask);
    const half2* add_val_h2 = reinterpret_cast<const half2*>(add_val);
    char2 mask_val = mask_c2[i];
    half2 one_or_zero_h2;
    half2 h2_scale = __float2half2_rn(scale);
    
    one_or_zero_h2.x = static_cast<bool>(mask_val.x);
    one_or_zero_h2.y = static_cast<bool>(mask_val.y);
    
    return __hadd2(__hmul2(__hmul2(x, one_or_zero_h2), h2_scale), add_val_h2[i]);
  }
  const uint8_t* mask;
  const half* add_val;
  float scale;
};

// biasAdd的输入两个，x.shape={rows, cols}, bias.shape={cols}, 所以需要在L59通过除余循环读取这cols个bias
template<typename FUNCTOR>
__global__ void FusedBiasAddCUDAKernelHalf2(FUNCTOR functor, const int elem_cnt,
                                        const int bias_size, const half* x, const half* bias,
                                        half* y) {
  const int h2_elem_cnt = elem_cnt / 2; // 读取的粒度由half变成了half2，那自然元素数量就少了一半
  const int h2_bias_size = bias_size / 2;
  const auto* x_h2 = reinterpret_cast<const half2*>(x); // 强转为向量指针后在L58读取
  const auto* bias_h2 = reinterpret_cast<const half2*>(bias);
  auto* y_h2 = reinterpret_cast<half2*>(y);
  // 保证有限线程数处理完所有数据
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < h2_elem_cnt;
       i += blockDim.x * gridDim.x){
    half2 x_i = __hadd2(x_h2[i], bias_h2[i % h2_bias_size]);
    
    y_h2[i] = functor.ComputeHalf2(x_i, i);
  }
}

template<typename FUNCTOR>
__global__ void FusedBiasAddCUDAKernelHalf(FUNCTOR functor, const int elem_cnt,
                                        const int bias_size, const half* x, const half* bias,
                                        half* y) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elem_cnt;
       i += blockDim.x * gridDim.x){
    half x_i = x[i] + bias[i % bias_size];
    y[i] = functor.Compute(x_i, i);
  }
}

__global__ void float2half(const float *in,__half *out,int size){
  int gtid = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = gtid;i <size;i += stride){
    out[i] = __float2half(in[i]);
  }
}

bool check(__half *out_gpu,float *out_cpu,int size){
  for (int i=0; i < size;i++){
    float out = __half2float(out_gpu[i]);
    if (fabs(out_cpu[i] - out) > 10e-6){
      printf("%d %f %f\n",i,out_cpu[i],out);
      return false;
    }
  }
  return true;
}

int main(){
    int ele_cnt = 25600000;
    float scale = 0.5;
    uint8_t* mask_tensor = new uint8_t[ele_cnt];
    float* add_val = new float[ele_cnt];
    for (int i = 0; i < ele_cnt; i++){
        mask_tensor[i] = (uint8_t)(i);
        add_val[i] = (float)(i % 32);
    }
    int bias_size = 10;
 
    __half *x = (__half*) malloc(sizeof(__half) * ele_cnt);
    __half *y = (__half*) malloc(sizeof(__half) * ele_cnt);
    __half *bias = (__half*) malloc(sizeof(__half) * bias_size);

    float *y_cpu = (float*) malloc(sizeof(float) * ele_cnt);

    for (int i = 0; i < ele_cnt; i++)
    {
      x[i] = (__half)(i % 32);
    }

    for (int i = 0;i <bias_size;i++){
      bias[i] = __int2half_rn(i);
    }

    __half *d_x, *d_y, *d_bias,*d_add_val_fp16;
    float *d_add_val_fp32;
    uint8_t *d_mask_tensor;
    cudaMalloc((void **)&d_x, ele_cnt * sizeof(__half));
    cudaMalloc((void **)&d_y, ele_cnt * sizeof(__half));
    cudaMalloc((void **)&d_bias, bias_size * sizeof(__half));

    cudaMalloc((void **)&d_mask_tensor, ele_cnt * sizeof(uint8_t));
    cudaMalloc((void **)&d_add_val_fp32, ele_cnt * sizeof(float));
    cudaMalloc((void **)&d_add_val_fp16, ele_cnt * sizeof(__half));

    cudaMemcpy(d_x, x, sizeof(__half) * ele_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(__half) * ele_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, sizeof(__half) * bias_size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_mask_tensor, mask_tensor, sizeof(uint8_t) * ele_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_add_val_fp32, add_val, sizeof(float) * ele_cnt, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxblocks = deviceProp.maxGridSize[0];
    int blockSize = 256;
    // int gridSize = std::min((ele_cnt + blockSize - 1) / blockSize, maxblocks);
    int gridSize = 1000;

    float2half<<<gridSize,blockSize>>> (d_add_val_fp32,d_add_val_fp16,ele_cnt);

    MaskScaleAndElementwiseAddFunctor<half> mask_scale_elementwise_add_func(d_mask_tensor, d_add_val_fp16, scale);
    
    MaskScaleAndElementwiseAddFunctor<float> mask_scale_elementwise_add_func_cpu(mask_tensor, add_val, scale);

    // FusedBiasAddCUDAKernelHalf2<<<gridSize ,blockSize>>>(mask_scale_elementwise_add_func, ele_cnt, bias_size, d_x, d_bias, d_y);

    FusedBiasAddCUDAKernelHalf<<<gridSize ,blockSize>>>(mask_scale_elementwise_add_func, ele_cnt, bias_size, d_x, d_bias, d_y);

    cudaMemcpy(y, d_y, sizeof(__half) * ele_cnt, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < ele_cnt; i++){
      float x_i = __half2float(x[i]);
      float bias_i = __half2float(bias[i%bias_size]);
      y_cpu[i] = mask_scale_elementwise_add_func_cpu.Compute((x_i + bias_i),i);
      // printf("(%f + %f) * %d * %f + %f = %f ? %f\n",x_i,bias_i,static_cast<bool>(mask_tensor[i]),scale,add_val[i],y_cpu[i],__half2float(y[i]));
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
    mask_tensor = nullptr;
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_bias);
    cudaFree(d_add_val_fp16);
    cudaFree(d_add_val_fp32);
    cudaFree(d_mask_tensor);
}
