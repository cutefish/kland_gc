#ifndef USER_HOSTS_H_
#define USER_HOSTS_H_

#include "mpi.h"

#include "support/TimingEvent.h"

struct RunEnv {
  int rank;
  int size;
  int dev_rank;
  float* dev_pCont;
  float* dev_pContMean;
  float* dev_pContVar;
  float* dev_pCorr;
  float* dev_pStack;
  float* host_pCont;
  float* host_pTemp;
  int num_temps;
  cuda::DeviceProperties dev_prop;
  support::TimingEventTable tevts;
};

/* initRunEnv()
 * initialize runtime environment. including:
 *  initialize mpi
 *  initialize gpu card
 *  calculate and allocate memory.
 *  initialize logging utils
 *  initialize timing utils
 */
RunEnv initRunEnv(int argc, char* argv[]);


#endif /* USER_HOSTS_H_ */
