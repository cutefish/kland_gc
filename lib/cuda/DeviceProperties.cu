#include <cuda_runtime_api.h>

#include "cuda/DeviceProperties.h"
#include "cuda/CudaErrCategory.h"

namespace cuda {

/*** DeviceProperties ***/

/* ctor/dtor */
DeviceProperties::DeviceProperties(const unsigned deviceID) {
  cudaDeviceProp* dev_prop = new cudaDeviceProp;
  checkCall(cudaGetDeviceProperties(dev_prop, deviceID));
  m_name = dev_prop->name;
  m_maxThreadsPerBlock = dev_prop->maxThreadsPerBlock;
  m_sharedMemPerBlock = dev_prop->sharedMemPerBlock;
  m_warpSize = dev_prop->warpSize;
  m_memPitch = dev_prop->memPitch;
  m_regsPerBlock = dev_prop->regsPerBlock;
  m_clockRate = dev_prop->clockRate;
  m_major = dev_prop->major;
  m_minor = dev_prop->minor;
  m_multiProcessorCount = dev_prop->multiProcessorCount;
  m_totalConstantMemory = dev_prop->totalConstMem;
  m_totalMemBytes = dev_prop->totalGlobalMem;
  m_textureAlign = dev_prop->textureAlignment;
}

/* getter/setter */
const std::string & name() const { return m_name; }
const int maxThreadsPerBlock() const { return m_maxThreadsPerBlock; }
//const Vector3<int> & getMaxBlockSize() const;
//const Vector3<int> & getMaxGridSize() const;
const size_t shmemSizePerBlock() const { return m_sharedMemPerBlock; }
const size_t constantMemorySize() const { return m_totalConstantMemory; }
const int warpSize() const { return m_warpSize; }
const int memoryPitch() const { return m_memPitch; }
const int numRegistersPerBlock() const { return m_regsPerBlock; }
const int clockRate() const { return m_clockRate; }
const size_t textureAlignment() const { return m_textureAlign; }
const size_t globalMemorySize() const { return m_totalMemBytes; }
const int capabilityMajor() const { return m_major; }
const int capabilityMinor() const { return m_minor; }
const int numMultiProcessors() const { return m_multiProcessorCount; }

} /* namespace cuda */
