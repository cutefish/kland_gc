#ifndef CUDA_CUDAERRCATEGORY_H_
#define CUDA_CUDAERRCATEGORY_H_

#include <cuda_runtime_api.h>

#include "support/exception/ErrorCode.h"

namespace cudaerr {

enum CudaErrEnum
{
  // from cuda/include/driver_types.h
  // mirror their order
  success                      = cudaSuccess,
  missing_configuration        = cudaErrorMissingConfiguration,
  memory_allocation            = cudaErrorMemoryAllocation,
  initialization_error         = cudaErrorInitializationError,
  launch_failure               = cudaErrorLaunchFailure,
  prior_launch_failure         = cudaErrorPriorLaunchFailure,
  launch_timeout               = cudaErrorLaunchTimeout,
  launch_out_of_resources      = cudaErrorLaunchOutOfResources,
  invalid_device_function      = cudaErrorInvalidDeviceFunction,
  invalid_configuration        = cudaErrorInvalidConfiguration,
  invalid_device               = cudaErrorInvalidDevice,
  invalid_value                = cudaErrorInvalidValue,
  invalid_pitch_value          = cudaErrorInvalidPitchValue,
  invalid_symbol               = cudaErrorInvalidSymbol,
  map_buffer_object_failed     = cudaErrorMapBufferObjectFailed,
  unmap_buffer_object_failed   = cudaErrorUnmapBufferObjectFailed,
  invalid_host_pointer         = cudaErrorInvalidHostPointer,
  invalid_device_pointer       = cudaErrorInvalidDevicePointer,
  invalid_texture              = cudaErrorInvalidTexture,
  invalid_texture_binding      = cudaErrorInvalidTextureBinding,
  invalid_channel_descriptor   = cudaErrorInvalidChannelDescriptor,
  invalid_memcpy_direction     = cudaErrorInvalidMemcpyDirection,
  address_of_constant_error    = cudaErrorAddressOfConstant,
  texture_fetch_failed         = cudaErrorTextureFetchFailed,
  texture_not_bound            = cudaErrorTextureNotBound,
  synchronization_error        = cudaErrorSynchronizationError,
  invalid_filter_setting       = cudaErrorInvalidFilterSetting,
  invalid_norm_setting         = cudaErrorInvalidNormSetting,
  mixed_device_execution       = cudaErrorMixedDeviceExecution,
  cuda_runtime_unloading       = cudaErrorCudartUnloading,
  unknown                      = cudaErrorUnknown,
  not_yet_implemented          = cudaErrorNotYetImplemented,
  memory_value_too_large       = cudaErrorMemoryValueTooLarge,
  invalid_resource_handle      = cudaErrorInvalidResourceHandle,
  not_ready                    = cudaErrorNotReady,
  insufficient_driver          = cudaErrorInsufficientDriver,
  set_on_active_process_error  = cudaErrorSetOnActiveProcess,
  no_device                    = cudaErrorNoDevice,
  ecc_uncorrectable            = cudaErrorECCUncorrectable,
  startup_failure              = cudaErrorStartupFailure
}; // end CudaErrEnum

} /* namespace cudaerr */

class CudaErrCategory : public support::ErrorCategory {
 public:
  typedef errc::CudaErrEnum type;

  virtual const char* name() const {
    return "cuda_error";
  }

  virtual std::string message(int ev) const {
    static const std::string unknown_err("Unknown error");
    const char* c_str = ::cudaGetErrorString(static_cast<cudaError_t>(ev));
    return c_str ? std::string(c_str) : unknown_err;
  }
  
};

namespace cuda {

static inline void checkCall(cudaError_t error, 
                             std::string message="") {
  if (error) {
    throw Exception(error,
                    getErrorCategory<CudaErrCategory>(), 
                    message);
  }
}

} /* namespace cuda */

#endif /* CUDA_CUDAERRCATEGORY_H_ */

