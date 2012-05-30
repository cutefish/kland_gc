#include <cmath>
#include <string>

#include "user/Config.h"
#include "user/Hosts.h"
#include "user/Works.h"
#include "user/UserErrCategory.h"
#include "user/UserIO.h"
#include "cuda/Runtime.h"
#include "support/Type.h"
#include "support/Exception.h"

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

void doLeaderLoop(Config cfg) {
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

void doWorkerLoop(Config cfg) {

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

/* doWork()
 * Actually do the work according to the configuration.
 */
void doWork(RunEnv env, Config cfg, TaskRange range) {
  std::vector<std::string>& cont_list = cfg.cont_list();
  std::vector<std::string>& temp_list = cfg.temp_list();
  std::vector<std::string>& ch_list = cfg.channel_list();

  //cont loop
  for (int ci = range.cont_start; ci < range.cont_end + 1; ++ci) {
    std::string cont_name = cont_list[ci];
    
    //track valid channels
    std::vector<unsigned> valid_channels;
    for (int ti = range.temp_start; ti < range.temp_end + 1; ++ti) {
      valid_channels.push_back(0);
    }

    //channel loop
    for (int chi = 0; chi < ch_list.size(); ++chi) {
      std::string chnl_name = ch_list[chi];
      //read cont data
      std::string path = cont_name + "/" + chnl_name;
      try {
        readContinuous(path, cfg, env.host_pCont);
      }
      catch (support::Exception e) {
        //logging
        //...
        continue;
      }
      //copy to device
      cuda::memcpyH2D(env.dev_pCont, env.host_pCont, 
                      cfg.cont_npts() * sizeof(float));
      //calculate mean and var
      getContMeanVar(env.dev_pCont, cfg.cont_npts(), cfg.temp_npts(),
                     env.dev_pContMean, env.dev_pContVar);

      //temp loop
      for (int ti = range.temp_start; ti < range.temp_end + 1; ++ti) {
        //read temp data
        std::string path = temp_name + "/" + chnl_name;
        try {
          readTemplate(path, cfg, host_pTemp);
        }
        catch (support::Exception e) {
          //logging
          //...
          continue;
        }
        //test snr
        path = temp_name + "/" + cfg.snr_name();
        float snr = readSNR(path, chnl_name);
        if (snr < cfg.snr_thr()) continue;

        valid_channels[ti] ++;
        //calculate mean and var
        float mean, var;
        getTempMeanVar(host_pTemp, cfg.temp_npts(), mean, var);
        //calculate correlation
        //... on ti 
        //stack correlation
        //... on ti
      }
    }

    //select
    for (int ti = range.temp_start; ti < range.temp_end + 1; ++ti) {
      //copy back
      cuda::memcpyD2H(env.host_pResult, 
                      env.dev_pStack + 
                      (ti - range.temp_start) * cfg.cont_npts(),
                      cfg.cont_npts() * sizeof(float));
      //get MAD
      float mad = getMAD(env.dev_pStack, cfg.cont_npts());

      //get path
      std::string path = "~/aaa"; //...
      std::ofstream ofs;
      ofs.open(path);
      throwError(ofs.is_open(), usererr::file_not_open, path);

      //select
      select(env.host_pResult, cfg.cont_npts(), mad, cfg.mad_ratio(),
             cfg.sample_rate(), valid_channels[ti], ofs);

    }
  }
}
