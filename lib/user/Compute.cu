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
  for (int i = offset; i < size - window_size; i += shift) {
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

/* clearDevData() */
void clearDevData(float* data ,unsigned size) {
  cuda::checkCall(cudaMemset((void*) data, 0, size * sizeof(float)));
}

/* cuda: CorrKernel() */
__global__ void CorrKernel(float* corr, float* temp, float* cont, 
                           unsigned cont_size, unsigned temp_size, 
                           float temp_mean, float temp_var,
                           float* cont_mean, float* cont_var) {
  unsigned shift = gridDim.x * blockDim.x;
  unsigned offset = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = offset; i < cont_size - temp_size; i += shift) {
    float ex = temp_mean;
    float ey = cont_mean[i];
    float sx = temp_var;
    float sy = cont_var[i];
    float sxy = 0;
    if (sx < 1.0e-5f || sy < 1.0e-5f) {
      corr[i] = 0;
    }
    else {
      for (int j = 0; j < temp_size; ++j) {
        float resx = temp[j] - ex;
        float resy = cont[i + j] - ey;
        sxy += resx * resy;
      }
      corr[i] = sxy / sqrtf(sx * sy);
    }
  }
}

/* calcCorr() */
void calcCorr(float* corr, float* temp, float* cont,
              unsigned cont_size, unsigned temp_size,
              float temp_mean, float temp_var,
              float* cont_mean, float* cont_var) {
  CorrKernel<<<GridSizeX, BlockSizeX>>>(corr, temp, cont, 
                                        cont_size, temp_size,
                                        temp_mean, temp_var, 
                                        cont_mean, cont_var);
  cuda::synchronize("calcCorr");
}

/* cuda: StackKernel() */
__global__ void StackKernel(float* corr, float* stack,
                            size_t corr_size, int stack_shift) {
  unsigned idx_shift = gridDim.x * blockDim.x;
  unsigned idx_offset = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = idx_offset; i < corr_size; i += idx_shift) {
//    float max = -2;
    float max = corr[i];
//    //if exists corr[i-1]
//    if (i > 0) max = corr[i - 1];
//    //max(corr[i-1], corr[i])
//    max = (max > corr[i]) ? max : corr[i];
//    //max(corr[i], corr[i+1])
//    if (i < corr_size - 1) max = (max > corr[i + 1]) ? max : corr[i + 1];
    //add to stack;
    if ((i - stack_shift) >= 0) stack[i - stack_shift] += max;
  }
}

/* stack() */
void stack(float* corr, float* stack, 
           size_t corr_size, int stack_shift) {
  StackKernel<<<GridSizeX, BlockSizeX>>>(corr, stack, corr_size, stack_shift);
  cuda::synchronize("stack");
}

/* cuda: AbsSubsMEDKernel() */
__global__ void AbsSubsMEDKernel(float* data, unsigned size, float* median) {
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
  AbsSubsMEDKernel<<<GridSizeX, BlockSizeX>>>(data, size, median);
  cuda::synchronize("AbsSubsMEDKernel");
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
  out << "mad value: " << mad / num_valid_channel << '\n';
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


