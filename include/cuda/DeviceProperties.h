#ifndef CUDA_DEVICEPROPERTIES_H_
#define CUDA_DEVICEPROPERTIES_H_

#include <string>

/*!\file DeviceProperties.h
 */

namespace cuda {

/*!\class DeviceProperties
 * \brief This class abstracts away a call to cudaGetDeviceProperties, and keeps
 * a variable-lifetime-persistent copy of the return value.
 */
class DeviceProperties
{
 public:
  /*** ctor/dtor ***/
  DeviceProperties(const unsigned deviceID);

  /*** member funcitons ***/
  /* getter/setter */
  const std::string & name() const;
  const int maxThreadsPerBlock() const;
  //const Vector3<int> & getMaxBlockSize() const;
  //const Vector3<int> & getMaxGridSize() const;
  const size_t shmemSizePerBlock() const;
  const size_t constantMemorySize() const;
  const int warpSize() const;
  const int memoryPitch() const;
  const int numRegistersPerBlock() const;
  const int clockRate() const;
  const size_t textureAlignment() const;
  const size_t globalMemorySize() const;
  const int capabilityMajor() const;
  const int capabilityMinor() const;
  const int numMultiProcessors() const;
 private:
  /// The name of the device.
  std::string m_name;
  /// The maximum number of threads in a single block.
  int m_maxThreadsPerBlock;
  /// The maximum number of threads in each dimension.
  //Vector3<int> maxThreadsDim;
  /// The maximum number of blocks in each dimension.
  //Vector3<int> maxGridSize;
  /// The total amount of shared memory available to a SM.
  int m_sharedMemPerBlock;
  /// The SIMD size of each SM.
  int m_warpSize;
  /// The pitch of global device memory.
  int m_memPitch;
  /// The total number of 32-bit registers per SM.
  int m_regsPerBlock;
  /// The frequency of the clock, in kHz.
  int m_clockRate;
  /// The device's compute capability major number.
  int m_major;
  /// The device's compute capability minor number.
  int m_minor;
  /// The number of SMs on the device.
  int m_multiProcessorCount;
  /// The total amount of constant memory available to a kernel.
  size_t m_totalConstantMemory;
  /// The total amount of global device memory.
  size_t m_totalMemBytes;
  /// The alignment requirements for textures in memory.
  size_t m_textureAlign;
};

} /* namespace cuda */

#endif /* CUDA_DEVICEPROPERTIES_H_ */
