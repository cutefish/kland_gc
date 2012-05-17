#include <cuda_runtime_api.h>

#include "Runtime.h"
#include "CudaErrCategory.h"
#include "support/bad_alloc.h"

namespace cuda {

inline int getDeviceCount() {
  int count;
  checkCall(cudaGetDeviceCount(&count), "getDeviceCount");
  return count;
}

inline void synchronize(const char* message) {
#if CUDART_VERSION >= 4000
  cudaError_t error = cudaDeviceSynchronize();
#else
  cudaError_t error = cudaThreadSynchronize();
#endif /* CUDART_VERSION */
  checkCall(error, std::string("synchronize: ") + message);
}

inline void* malloc(const size_t n, const char* message="") {
  void* ret = 0;

  cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&ret), n);

  if (error) {
    throw support::bad_alloc(
        (getErrorCategory<CudaErrCategory>().message(error) + 
         message).c_str());
}

inline void memcpyH2D(void* dst, const void* src, const size_t count,
                      const char* message="") {
  cudaError_t error = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
  checkCall(error, std::string("memcpyH2D: ") + message);
}

inline void memcpyD2H(void* dst, const void* src, const size_t count,
                      const char* message="") {
  cudaError_t error = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
  checkCall(error, std::string("memcpyD2H: ") + message);
}

} /* namespace cuda */
