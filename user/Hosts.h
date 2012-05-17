#ifndef USER_MPIHOSTS_H_
#define USER_MPIHOSTS_H_

#include "mpi.h"
#include <vector>

struct RunEnv {
  int rank;
  int size;
  int dev_rank;
};

/* initRunEnv()
 * initialize mpi and get environment
 */
RunEnv initRunEnv(int argc, char* argv[]);


#endif /* USER_MPIHOSTS_H_ */
