#ifndef CUDA_RUNTIME_H_
#define CUDA_RUNTIME_H_

namespace cuda {

int getDeviceCount();

void setDevice(int rank);

void synchronize(const char* message);

void* malloc(const size_t n, const char* message="");

void free(void* ptr, const char* message="");

inline void memcpyH2D(void* dst, const void* src, const size_t count,
                      const char* message="");

inline void memcpyD2H(void* dst, const void* src, const size_t count,
                      const char* message="");

inline void memcpyD2D(void* dst, const void* src, const size_t count,
                      const char* message="");

inline void memcpyH2H(void* dst, const void* src, const size_t count,
                      const char* message="");

} /* namespace cuda */

#endif /* CUDA_RUNTIME_H_ */
