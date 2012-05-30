#ifndef USER_HOSTS_H_
#define USER_HOSTS_H_

#include "mpi.h"
#include <map>

struct RunEnv {
  int dev_rank;
  float* dev_pCont;
  float* dev_pContMean;
  float* dev_pContVar;
  float* host_pCont;
  float* host_pTemp;
  float* dev_pCorr;
  float* dev_pStack;
  float* host_pResult;
};

/* initRunEnv()
 * initialize runtime environment.
 */
RunEnv initRunEnv(int argc, char* argv[]);


#endif /* USER_HOSTS_H_ */
