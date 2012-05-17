#include <cmath>

#include "user/Partition.h"

std::list<TaskRange> genWorkList(int num_temp, int num_cont,
                                 int tseg_size, int cseg_size) {
  std::list<TaskRange> ret;
  for (int i = 0; i < num_temp; i += tseg_size) {
    for (int j = 0; j < num_cont; j += cseg_size) {
      TaskRange range;
      range.temp_start = i;
      range.temp_end = (i + tseg_size - 1 < num_temp - 1) ?
          (i + tseg_size - 1) : (num_temp - 1);
      range.cont_start = j;
      range.cont_end = (j + cseg_size - 1 < num_cont - 1) ?
          (j + cseg_size - 1) : (num_cont - 1);
      ret.push_back(range);
    }
  }
  return ret;
}

std::list<TaskRange> partition(int num_proc, 
                                 int num_temp, int num_cont, 
                                 int temp_npts, int cont_npts) {
  //solve the problem
  float opt_temp = sqrt(num_temp * num_cont * cont_npts /
                        num_proc / temp_npts);
  int tseg_size = (opt_temp > num_temp) ? num_temp : (int)opt_temp;
  int cseg_size = num_temp * num_cont / num_proc / tseg_size;

  //generate work list
  return genWorkList(num_temp, num_cont, tseg_size, cseg_size);
}
