#include <cmath>

#include "Config.h"
#include "Hosts.h"
#include "Works.h"
#include "UserErrCategory.h"
#include "cuda/Runtime.h"
#include "support/Type.h"

enum MsgTags {
  WorkTag,
  ReqTag,
  DoneTag,
};

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

void doLeaderLoop(RunEnv env, Config cfg) {
  MPI_Status mpi_status;
  int out_buf[sizeof(TaskRange) / sizeof(int)];

  //all works
  std::list<TaskRange> works = partition(env.size,
                                         cfg.temp_list().size(),
                                         cfg.cont_list().size(),
                                         cfg.temp_npts(), cfg.cont_npts());

  while (!works.empty()) {
    //wait for receving a request.
    MPI_Recv(0, 0, MPI_INT, MPI_ANY_SOURCE, ReqTag, MPI_COMM_WORLD, &mpi_status);

    TaskRange work = works.front();
    works.pop_front();

    out_buf[0] = work.temp_start;
    out_buf[1] = work.temp_end;
    out_buf[2] = work.cont_start;
    out_buf[3] = work.cont_end;

    MPI_Send(out_buf, 4, MPI_INT, mpi_status.MPI_SOURCE, WorkTag, MPI_COMM_WORLD);
  }

  //all work done
  for (int i = 0; i < mpi_size; ++i) {
    MPI_Send(0, 0, MPI_INT, i, DoneTag, MPI_COMM_WORLD);
  }
}

void doWorkerLoop(RunEnv env, Config cfg) {

  MPI_Status mpi_status;
  int out_buf[sizeof(TaskRange) / sizeof(int)];

  while(1) {
    //send a work request to leader
    MPI_Send(0, 0, MPI_INT, 0, ReqTag, MPI_COMM_WORLD);

    //wait for a message
    MPI_Recv(out_buf, 4, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &mpi_status);

    if (mpi_status.MPI_TAG == DoneTag) break;

    //do work
    std::cout << "Worker: " << rank << "; "
        << "ts: " << out_buf[0] << "; "
        << "te: " << out_buf[1] << "; "
        << "cs: " << out_buf[2] << "; "
        << "ce: " << out_buf[3] << '\n';
    sleep(3);
  }
}
