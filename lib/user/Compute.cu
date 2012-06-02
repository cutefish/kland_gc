#include <cuda_runtime_api.h>
#include <iomanip>
#include <sstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>

#include "cuda/Runtime.h"
#include "cuda/CudaErrCategory.h"
#include "user/Compute.h"

static const unsigned BlockSizeX = 512;
static const unsigned GridSizeX = 56;

/* getTempMeanVar() */
void getTempMeanVar(float* data, unsigned size, float& mean, float& var) {
  mean = 0;
  for (int i = 0; i < size; ++i) {
    mean += data[i];
  }
  mean = mean / size;
  var = 0;
  for (int i = 0; i < size; ++i) {
    float res = data[i] - mean;
    var += res * res;
  }
}

/* cuda: ContMeanVarKernel() */
__global__ void ContMeanVarKernel(float* data, unsigned size, unsigned window_size,
                             float* mean, float* var) {
  unsigned shift = gridDim.x * blockDim.x;
  unsigned offset = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = offset; i < size; i += shift) {
    float m = 0;
    for (int j = 0; j < window_size; ++j) {
      m += data[i + j];
    }
    m = m / window_size;
    mean[i] = m;
    float v = 0;
    for (int j = 0; j < window_size; ++j) {
      float res = data[i + j] - m;
      v += res * res;
    }
    var[i] = v;
  }
}

/* getContMeanVar() */
void getContMeanVar(float* data, unsigned size, unsigned window_size,
                    float* mean, float* var) {
  ContMeanVarKernel<<<GridSizeX, BlockSizeX>>>(data, size, window_size,
                                               mean, var);
  cuda::synchronize("getContMeanVar");
}

/* clearStack() */
void clearStack(float* data ,unsigned size) {
  cuda::checkCall(cudaMemset((void*) data, 0, size * sizeof(float)));
}

/* cuda: AbsSubsMADKernel() */
__global__ void AbsSubsMADKernel(float* data, unsigned size, float* median) {
  unsigned shift = gridDim.x * blockDim.x;
  unsigned offset = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = offset; i < size; i += shift) {
    data[i] = abs(data[i] - (*median));
  }
}

/* getMAD() */
float getMAD(float* data, unsigned size) {
  //sort first
  thrust::device_ptr<float> dev_data(data);
  thrust::sort(dev_data, dev_data + size);
  cuda::synchronize("thrust_sort1");
  //get median
  float* median = reinterpret_cast<float*>(cuda::malloc(sizeof(float)));
  cuda::memcpyD2D(median, data + size / 2, sizeof(float), "getMAD");
  //abs(data[i] - median)
  AbsSubsMADKernel<<<GridSizeX, BlockSizeX>>>(data, size, median);
  cuda::synchronize("AbsSubsMADKernel");
  //sort again
  thrust::sort(dev_data, dev_data + size);
  cuda::synchronize("thrust_sort2");
  //get mad result
  float mad;
  cuda::memcpyD2H((void*)&mad, data + size / 2, sizeof(float), "getMAD");
  //clean up
  cuda::free(median);
  return mad;
}

/* select() */
void select(float* data, unsigned size, 
            float mad, float ratio, 
            float sample_rate, unsigned num_valid_channel, 
			std::ofstream& out) {
  out << "mad value: " << mad << '\n';
  for (int i = 0; i < size; ++i) {
    if (data[i] > (mad * ratio)) {
      std::stringstream ss;
      float time = static_cast<float>(i) / sample_rate;
      float corr = data[i] / num_valid_channel;
      ss << std::fixed << std::setprecision(2) << time << '\t'
          << std::setprecision(5) << corr;
      out << ss.str() << '\n';
    }
  }
}


