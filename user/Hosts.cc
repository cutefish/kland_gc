#include <cmath>

#include "Hosts.h"
#include "UserErrCategory.h"
#include "cuda/Runtime.h"
#include "support/Type.h"

/* initRunEnv() */
RunEnv initRunEnv(int& argc, char**& argv) {
 int errno = MPI_Init(&argc, &argv);
 user::throwError(errno == MPI_SUCCESS, 
                  usererr::mpi_error,
                  "errorno: " + support::Type2String<int>(errno));
 RunEnv env;
 MPI_Comm_rank(MPI_COMM_WORLD, &env.rank);
 MPI_Comm_size(MPI_COMM_WORLD, &env.size);
 env.dev_rank = env.rank % cuda::getDeviceCount();
 return env;
}

