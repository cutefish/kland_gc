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
  m_pciDeviceID = dev_prop->pciBusID * 10000 + 
      dev_prop->pciDeviceID * 100 + dev_prop->pciDomainID;
}

/* getter/setter */
const std::string & DeviceProperties::name() const { 
	return m_name; 
}
int DeviceProperties::maxThreadsPerBlock() const { 
	return m_maxThreadsPerBlock; 
}
//const Vector3<int> & getMaxBlockSize() const;
//const Vector3<int> & getMaxGridSize() const;
size_t DeviceProperties::shmemSizePerBlock() const { 
	return m_sharedMemPerBlock; 
}
size_t DeviceProperties::constantMemorySize() const { 
	return m_totalConstantMemory; 
}
int DeviceProperties::warpSize() const { 
	return m_warpSize; 
}
int DeviceProperties::memoryPitch() const { 
	return m_memPitch; 
}
int DeviceProperties::numRegistersPerBlock() const { 
	return m_regsPerBlock; 
}
int DeviceProperties::clockRate() const { 
	return m_clockRate; 
}
size_t DeviceProperties::textureAlignment() const { 
	return m_textureAlign; 
}
size_t DeviceProperties::globalMemorySize() const { 
	return m_totalMemBytes; 
}
int DeviceProperties::capabilityMajor() const { 
	return m_major; 
}
int DeviceProperties::capabilityMinor() const { 
	return m_minor; 
}
int DeviceProperties::numMultiProcessors() const { 
	return m_multiProcessorCount; 
}
int DeviceProperties::pciDeviceID() const {
  return m_pciDeviceID;
}

} /* namespace cuda */
