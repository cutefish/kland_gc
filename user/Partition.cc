#include <cmath>

#include "Partition.h"

std::vector<WorkRange> partition(int num_proc, 
                                 int num_temp, int num_cont, 
                                 int temp_npts, int cont_npts) {
  //solve the problem
  float opt_temp = sqrt(num_temp * num_cont * cont_npts /
                        num_proc / temp_npts);
  int tseg_size = (opt_temp > num_temp) ? num_temp : (int)opt_temp;
  int cseg_size = num_temp * num_cont / num_proc / tseg_size;

  //partition
  //be careful with the boundary
  int num_tseg = num_temp / tseg_size;
  for (int i = 0; i < num_proc; ++i) {
  }

}
