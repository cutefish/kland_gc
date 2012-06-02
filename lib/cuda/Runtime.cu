#include <cuda_runtime_api.h>

#include "cuda/Runtime.h"
#include "cuda/CudaErrCategory.h"
#include "support/bad_alloc.h"

namespace cuda {

/* getDeviceCount() */
int getDeviceCount() {
  int count;
  checkCall(cudaGetDeviceCount(&count), "getDeviceCount");
  return count;
}

/* setDevice() */
void setDevice(int rank) {
  checkCall(cudaSetDevice(rank), "setDevice");
}

/* synchronize() */
void synchronize(const char* message) {
#if CUDART_VERSION >= 4000
  cudaError_t error = cudaDeviceSynchronize();
#else
  cudaError_t error = cudaThreadSynchronize();
#endif /* CUDART_VERSION */
  checkCall(error, std::string("synchronize: ") + message);
}

/* malloc() */
void* malloc(const size_t n, const char* message) {
  void* ret = 0;

  cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&ret), n);

  if (error) {
    throw support::bad_alloc(
        (support::getErrorCategory<CudaErrCategory>().message(error) + 
         message).c_str());
  }
  return ret;
}

/* free() */
void free(void* ptr, const char* message) {
  cudaError_t error = cudaFree(ptr);
  if (error) {
    throw support::Exception(error, 
                             support::getErrorCategory<CudaErrCategory>(),
                             message);
  }
}

/* memcpy() */
void memcpyH2D(void* dst, const void* src, const size_t count,
                      const char* message) {
  cudaError_t error = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
  checkCall(error, std::string("memcpyH2D: ") + message);
}


void memcpyD2H(void* dst, const void* src, const size_t count,
                      const char* message) {
  cudaError_t error = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
  checkCall(error, std::string("memcpyD2H: ") + message);
}

void memcpyD2D(void* dst, const void* src, const size_t count,
                      const char* message) {
  cudaError_t error = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
  checkCall(error, std::string("memcpyD2D: ") + message);
}

void memcpyH2H(void* dst, const void* src, const size_t count,
                      const char* message) {
  cudaError_t error = cudaMemcpy(dst, src, count, cudaMemcpyHostToHost);
  checkCall(error, std::string("memcpyH2H: ") + message);
}
} /* namespace cuda */
