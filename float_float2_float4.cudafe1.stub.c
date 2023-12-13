#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "float_float2_float4.fatbin.c"
typedef unsigned long _ZN6thrust6detail18contiguous_storageIfNS_16device_allocatorIfEEE9size_typeE;
typedef thrust::cuda_cub::__parallel_for::ParallelForAgent< ::thrust::cuda_cub::__uninitialized_fill::functor< ::thrust::device_ptr<float> , float> , unsigned long>  _ZN6thrust8cuda_cub14__parallel_for16ParallelForAgentINS0_20__uninitialized_fill7functorINS_10device_ptrIfEEfEEmEE;
typedef thrust::cuda_cub::__uninitialized_fill::functor< ::thrust::device_ptr<float> , float>  _ZN6thrust8cuda_cub20__uninitialized_fill7functorINS_10device_ptrIfEEfEE;
extern void __device_stub__Z9add_floatPfS_S_j(float *, float *, float *, unsigned);
extern void __device_stub__Z10add_float2P6float2S0_S0_j(struct float2 *, struct float2 *, struct float2 *, unsigned);
extern void __device_stub__Z10add_float4P6float4S0_S0_j(struct float4 *, struct float4 *, struct float4 *, unsigned);
static void __device_stub__ZN6thrust8cuda_cub4core13_kernel_agentINS0_14__parallel_for16ParallelForAgentINS0_20__uninitialized_fill7functorINS_10device_ptrIfEEfEEmEES9_mEEvT0_T1_( _ZN6thrust8cuda_cub20__uninitialized_fill7functorINS_10device_ptrIfEEfEE&, _ZN6thrust6detail18contiguous_storageIfNS_16device_allocatorIfEEE9size_typeE);
static void __device_stub__ZN3cub17CUB_200200_860_NS11EmptyKernelIvEEvv(void);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z9add_floatPfS_S_j(float *__par0, float *__par1, float *__par2, unsigned __par3){__cudaLaunchPrologue(4);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaLaunch(((char *)((void ( *)(float *, float *, float *, unsigned))add_float)));}
# 28 "./float_float2_float4.cu"
void add_float( float *__cuda_0,float *__cuda_1,float *__cuda_2,unsigned __cuda_3)
# 28 "./float_float2_float4.cu"
{__device_stub__Z9add_floatPfS_S_j( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 65 "./float_float2_float4.cu"
}
# 1 "float_float2_float4.cudafe1.stub.c"
void __device_stub__Z10add_float2P6float2S0_S0_j( struct float2 *__par0,  struct float2 *__par1,  struct float2 *__par2,  unsigned __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(struct float2 *, struct float2 *, struct float2 *, unsigned))add_float2))); }
# 70 "./float_float2_float4.cu"
void add_float2( struct float2 *__cuda_0,struct float2 *__cuda_1,struct float2 *__cuda_2,unsigned __cuda_3)
# 70 "./float_float2_float4.cu"
{__device_stub__Z10add_float2P6float2S0_S0_j( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 95 "./float_float2_float4.cu"
}
# 1 "float_float2_float4.cudafe1.stub.c"
void __device_stub__Z10add_float4P6float4S0_S0_j( struct float4 *__par0,  struct float4 *__par1,  struct float4 *__par2,  unsigned __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(struct float4 *, struct float4 *, struct float4 *, unsigned))add_float4))); }
# 100 "./float_float2_float4.cu"
void add_float4( struct float4 *__cuda_0,struct float4 *__cuda_1,struct float4 *__cuda_2,unsigned __cuda_3)
# 100 "./float_float2_float4.cu"
{__device_stub__Z10add_float4P6float4S0_S0_j( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 119 "./float_float2_float4.cu"
}
# 1 "float_float2_float4.cudafe1.stub.c"
static void __device_stub__ZN6thrust8cuda_cub4core13_kernel_agentINS0_14__parallel_for16ParallelForAgentINS0_20__uninitialized_fill7functorINS_10device_ptrIfEEfEEmEES9_mEEvT0_T1_(  _ZN6thrust8cuda_cub20__uninitialized_fill7functorINS_10device_ptrIfEEfEE&__par0,  _ZN6thrust6detail18contiguous_storageIfNS_16device_allocatorIfEEE9size_typeE __par1) {  __cudaLaunchPrologue(2); __cudaSetupArg(__par0, 0UL); __cudaSetupArgSimple(__par1, 16UL); __cudaLaunch(((char *)((void ( *)( _ZN6thrust8cuda_cub20__uninitialized_fill7functorINS_10device_ptrIfEEfEE, _ZN6thrust6detail18contiguous_storageIfNS_16device_allocatorIfEEE9size_typeE))thrust::cuda_cub::core::_kernel_agent< ::thrust::cuda_cub::__parallel_for::ParallelForAgent< ::thrust::cuda_cub::__uninitialized_fill::functor< ::thrust::device_ptr<float> , float> , unsigned long> ,  ::thrust::cuda_cub::__uninitialized_fill::functor< ::thrust::device_ptr<float> , float> , unsigned long> ))); }namespace thrust{
namespace cuda_cub{
namespace core{

template<> __specialization_static void __wrapper__device_stub__kernel_agent< ::thrust::cuda_cub::__parallel_for::ParallelForAgent< ::thrust::cuda_cub::__uninitialized_fill::functor< ::thrust::device_ptr<float> , float> , unsigned long> , ::thrust::cuda_cub::__uninitialized_fill::functor< ::thrust::device_ptr<float> , float> ,unsigned long>(  _ZN6thrust8cuda_cub20__uninitialized_fill7functorINS_10device_ptrIfEEfEE &__cuda_0,_ZN6thrust6detail18contiguous_storageIfNS_16device_allocatorIfEEE9size_typeE &__cuda_1){__device_stub__ZN6thrust8cuda_cub4core13_kernel_agentINS0_14__parallel_for16ParallelForAgentINS0_20__uninitialized_fill7functorINS_10device_ptrIfEEfEEmEES9_mEEvT0_T1_( ( _ZN6thrust8cuda_cub20__uninitialized_fill7functorINS_10device_ptrIfEEfEE &)__cuda_0,(_ZN6thrust6detail18contiguous_storageIfNS_16device_allocatorIfEEE9size_typeE &)__cuda_1);}}}}
static void __device_stub__ZN3cub17CUB_200200_860_NS11EmptyKernelIvEEvv(void) {  __cudaLaunchPrologue(1); __cudaLaunch(((char *)((void ( *)(void))cub::CUB_200200_860_NS::EmptyKernel<void> ))); }namespace cub{
inline namespace CUB_200200_860_NS{

template<> __specialization_static void __wrapper__device_stub_EmptyKernel<void>(void){__device_stub__ZN3cub17CUB_200200_860_NS11EmptyKernelIvEEvv();}}}
static void __nv_cudaEntityRegisterCallback( void **__T154) {  __nv_dummy_param_ref(__T154); __nv_save_fatbinhandle_for_managed_rt(__T154); __cudaRegisterEntry(__T154, ((void ( *)(void))cub::CUB_200200_860_NS::EmptyKernel<void> ), _ZN3cub17CUB_200200_860_NS11EmptyKernelIvEEvv, (-1)); __cudaRegisterEntry(__T154, ((void ( *)( _ZN6thrust8cuda_cub20__uninitialized_fill7functorINS_10device_ptrIfEEfEE, _ZN6thrust6detail18contiguous_storageIfNS_16device_allocatorIfEEE9size_typeE))thrust::cuda_cub::core::_kernel_agent< ::thrust::cuda_cub::__parallel_for::ParallelForAgent< ::thrust::cuda_cub::__uninitialized_fill::functor< ::thrust::device_ptr<float> , float> , unsigned long> ,  ::thrust::cuda_cub::__uninitialized_fill::functor< ::thrust::device_ptr<float> , float> , unsigned long> ), _ZN6thrust8cuda_cub4core13_kernel_agentINS0_14__parallel_for16ParallelForAgentINS0_20__uninitialized_fill7functorINS_10device_ptrIfEEfEEmEES9_mEEvT0_T1_, ( ::thrust::cuda_cub::__parallel_for::PtxPolicy<(int)256, (int)2> ::BLOCK_THREADS)); __cudaRegisterEntry(__T154, ((void ( *)(struct float4 *, struct float4 *, struct float4 *, unsigned))add_float4), _Z10add_float4P6float4S0_S0_j, (-1)); __cudaRegisterEntry(__T154, ((void ( *)(struct float2 *, struct float2 *, struct float2 *, unsigned))add_float2), _Z10add_float2P6float2S0_S0_j, (-1)); __cudaRegisterEntry(__T154, ((void ( *)(float *, float *, float *, unsigned))add_float), _Z9add_floatPfS_S_j, (-1)); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608026thrust6system6detail10sequential3seqE,::thrust::system::detail::sequential::seq), 0, 1UL, 0, 0); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608026thrust8cuda_cub3parE,::thrust::cuda_cub::par), 0, 1UL, 0, 0); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608026thrust8cuda_cub10par_nosyncE,::thrust::cuda_cub::par_nosync), 0, 1UL, 0, 0); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608026thrust12placeholders2_1E,::thrust::placeholders::_1), 0, 1UL, 0, 0); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608026thrust12placeholders2_2E,::thrust::placeholders::_2), 0, 1UL, 0, 0); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608026thrust12placeholders2_3E,::thrust::placeholders::_3), 0, 1UL, 0, 0); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608026thrust12placeholders2_4E,::thrust::placeholders::_4), 0, 1UL, 0, 0); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608026thrust12placeholders2_5E,::thrust::placeholders::_5), 0, 1UL, 0, 0); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608026thrust12placeholders2_6E,::thrust::placeholders::_6), 0, 1UL, 0, 0); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608026thrust12placeholders2_7E,::thrust::placeholders::_7), 0, 1UL, 0, 0); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608026thrust12placeholders2_8E,::thrust::placeholders::_8), 0, 1UL, 0, 0); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608026thrust12placeholders2_9E,::thrust::placeholders::_9), 0, 1UL, 0, 0); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608026thrust12placeholders3_10E,::thrust::placeholders::_10), 0, 1UL, 0, 0); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608026thrust3seqE,::thrust::seq), 0, 1UL, 0, 0); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608024cuda3std3__48in_placeE,::cuda::std::__4::in_place), 0, 1UL, 0, 0); __cudaRegisterVariable(__T154, __shadow_var(_ZN53_INTERNAL_e58ba72d_22_float_float2_float4_cu_7d2608024cuda3std6ranges3__45__cpo4swapE,::cuda::std::ranges::__4::__cpo::swap), 0, 1UL, 0, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
