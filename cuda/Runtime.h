#ifndef CUDA_RUNTIME_H_
#define CUDA_RUNTIME_H_

namespace cuda {

int getDeviceCount();

void synchronize(const char* message);

void* malloc(const size_t n);

} /* namespace cuda */

#endif /* CUDA_RUNTIME_H_ */
