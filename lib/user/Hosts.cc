#include <cmath>
#include <list>
#include <string>
#include <sys/stat.h>
#include <sstream>
#include <iostream>

#include "cuda/Runtime.h"
#include "support/Type.h"
#include "support/Exception.h"
#include "support/Logging.h"
#include "support/Type.h"
#include "support/StringUtils.h"
#include "user/Config.h"
#include "user/Compute.h"
#include "user/Hosts.h"
#include "user/UserErrCategory.h"
#include "user/UserIO.h"

enum MsgTags {
  WorkTag,
  ReqTag,
  DoneTag,
};

struct TaskRange {
  int temp_start;
  int temp_end;
  int cont_start;
  int cont_end;
};

/* init() */
RunEnv init(Config cfg) {
  RunEnv env;
  //initialize mpi
  int error = MPI_Init(NULL, NULL);
  user::throwError(error == MPI_SUCCESS, 
                   usererr::mpi_call_error,
                   "errorno: " + support::Type2String<int>(errno));
  MPI_Comm_rank(MPI_COMM_WORLD, &env.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &env.size);
  env.dev_rank = env.rank % cuda::getDeviceCount();

  //initialize gpu card
  env.dev_prop = new cuda::DeviceProperties(env.dev_rank);
  cuda::setDevice(env.dev_rank);

  //calculate and allocate memory
  size_t capacity = env.dev_prop->globalMemorySize() * 0.8;
  size_t left;
  size_t temp_req, cont_req;
  ///device: one continuous waveform at a time
  temp_req = cfg.temp_npts() * sizeof(float);
  cont_req = cfg.cont_npts() * sizeof(float);
  left = capacity -temp_req - cont_req * 5; // an extra for sorting
  env.num_temps = left / cont_req; //all the left for stack
  env.dev_pTemp = reinterpret_cast<float*>(cuda::malloc(temp_req));
  env.dev_pCont = reinterpret_cast<float*>(cuda::malloc(cont_req));
  env.dev_pContMean = reinterpret_cast<float*>(cuda::malloc(cont_req));
  env.dev_pContVar = reinterpret_cast<float*>(cuda::malloc(cont_req));
  env.dev_pCorr = reinterpret_cast<float*>(cuda::malloc(cont_req));
  env.dev_pStack = reinterpret_cast<float*>(cuda::malloc(env.num_temps * 
                                                         cont_req));
  ///host
  env.host_pCont = new float[cfg.cont_npts()];
  env.host_pTemp = new float[cfg.temp_npts()];

  //initialize logging utils
  support::LogSys::init();
  support::Logger& user_log = support::LogSys::newLogger("userLog");
  support::Logger& perf_log = support::LogSys::newLogger("perfLog");
  user_log.setLevel(support::DEBUG);
  perf_log.setLevel(support::DEBUG);
  std::string log_path;
  //log to <work_dir>/log_<timestamp>/<launch_id>/userLog<rank>
  log_path = cfg.log_root() + "/userLog" + support::Type2String<int>(env.rank);
  support::HandlerRef user_hdlr = support::FileHandler::create(log_path);
  user_hdlr.setLevel(support::DEBUG);
  user_log.addHandler(user_hdlr);
  //log to <work_dir>/log_<timestamp>/<launch_id>/perfLog<rank>
  log_path = cfg.log_root() + "/perfLog" + support::Type2String<int>(env.rank);
  support::HandlerRef perf_hdlr = support::FileHandler::create(log_path);
  perf_hdlr.setLevel(support::DEBUG);
  perf_log.addHandler(perf_hdlr);

  //initialize timing utils
  support::TimingSys::init();
  support::TimingSys::newEvent("totalTime");
  support::TimingSys::newEvent("readTemplate");
  support::TimingSys::newEvent("calcTemplateMeanVar");
  support::TimingSys::newEvent("readContinuous");
  support::TimingSys::newEvent("copyContinuous");
  support::TimingSys::newEvent("calcContinuousMeanVar");
  support::TimingSys::newEvent("calcCorr");
  support::TimingSys::newEvent("stackCorr");
  support::TimingSys::newEvent("calcMAD");
  support::TimingSys::newEvent("select");

  perf_log.info("Done initialization.\n");
  perf_log.info("Batch temp size: " + support::Type2String<int>(env.num_temps) + '\n');

  return env;
}

/* finalize() */
void finalize(RunEnv env) {
  //finalize mpi
  int errer = MPI_Finalize();
  user::throwError(errer == MPI_SUCCESS, 
                   usererr::mpi_call_error,
                   "errorno: " + support::Type2String<int>(errno));
  
  //free pointers
  cuda::free(env.dev_pTemp);
  cuda::free(env.dev_pCont);
  cuda::free(env.dev_pContMean);
  cuda::free(env.dev_pContVar);
  cuda::free(env.dev_pCorr);
  cuda::free(env.dev_pStack);
  delete [] env.host_pCont;
  delete [] env.host_pTemp;

  //log performance stats
  for (support::TimingSys::iterator it = support::TimingSys::begin();
       it != support::TimingSys::end(); ++it) {
    if ((*it).name() == "totalTime")
      support::LogSys::getLogger("perfLog").info(
          "Total time: " + support::Time2String(
              (*it).tot_dur(), "%TH:%Tm:%Ts\n\n"));
    else
      support::LogSys::getLogger("perfLog").info(
          (*it).name() + "\n" + 
          "total: " + support::Time2String((*it).run_dur(), "%TH h:%Tm m:%Ts s\n") + 
          "ave: " + support::Time2String((*it).ave_dur(), "%Ti ms\n") + 
          "num: " + support::Type2String<int>((*it).num_pauses()) + 
          "\n\n");

  }

  //finalize logging
  support::LogSys::finalize();

  //finalize timing
  support::TimingSys::finalize();
}

std::list<TaskRange> partition(int num_proc, int tseg_size, 
                               int num_temp, int num_cont) {
  int cseg_size = num_cont / num_proc;
  tseg_size = ((num_temp / num_proc) < tseg_size) ? 
      (num_temp / num_proc) : tseg_size;

  cseg_size = (cseg_size == 0) ? 1 : cseg_size;
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

void doLeaderLoop(RunEnv env, Config cfg) {
  MPI_Status mpi_status;
  int out_buf[sizeof(TaskRange) / sizeof(int)];

  //all works
  std::list<TaskRange> works = partition(env.size, env.num_temps, 
                                         cfg.temp_list().size(),
                                         cfg.cont_list().size());
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
  for (int i = 0; i < env.size; ++i) {
    MPI_Send(0, 0, MPI_INT, i, DoneTag, MPI_COMM_WORLD);
  }
}

//forward declaration
void doWork(RunEnv env, Config cfg, TaskRange range);

void doWorkerLoop(RunEnv env, Config cfg) {

  support::TimingSys::startEvent("totalTime");

  MPI_Status mpi_status;
  int in_buf[sizeof(TaskRange) / sizeof(int)];

  while(1) {
    //send a work request to leader
    MPI_Send(0, 0, MPI_INT, 0, ReqTag, MPI_COMM_WORLD);

    //wait for a message
    MPI_Recv(in_buf, 4, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &mpi_status);

    if (mpi_status.MPI_TAG == DoneTag) break;

    //do work
    TaskRange work;
    work.temp_start = in_buf[0];
    work.temp_end = in_buf[1];
    work.cont_start = in_buf[2];
    work.cont_end = in_buf[3];
    support::LogSys::getLogger("perfLog").debug(
        "Received work: " + 
        support::Type2String<int>(in_buf[0]) + ", " + 
        support::Type2String<int>(in_buf[1]) + ", " + 
        support::Type2String<int>(in_buf[2]) + ", " + 
        support::Type2String<int>(in_buf[3]) + "\n");
    doWork(env, cfg, work);
  }

  support::TimingSys::endEvent("totalTime");
}

/* doWork()
 * Actually do the work according to the configuration.
 */
void doWork(RunEnv env, Config cfg, TaskRange range) {
  const std::vector<std::string>& cont_list = cfg.cont_list();
  const std::vector<std::string>& temp_list = cfg.temp_list();
  const std::vector<std::string>& ch_list = cfg.channel_list();

  //only log temp info on first loop
  bool first_temp_loop = true;

  //cont loop
  for (int ci = range.cont_start; ci < range.cont_end + 1; ++ci) {
    std::string cont_name = cont_list[ci];

    //track valid channels
    std::vector<int> valid_channels;
    for (int ti = range.temp_start; ti < range.temp_end + 1; ++ti) {
      valid_channels.push_back(0);
    }

    //data config
    DataConfig cont_cfg, temp_cfg;
    size_t max_cont_npts = 0;

    //clear stack
    clearDevData(env.dev_pStack, 
                 env.num_temps * cfg.cont_npts());
    
    //channel loop
    for (int chi = 0; chi < ch_list.size(); ++chi) {
      //clear cont data
      clearDevData(env.dev_pCont, cfg.cont_npts());
      clearDevData(env.dev_pContMean, cfg.cont_npts());
      clearDevData(env.dev_pContVar, cfg.cont_npts());

      std::string chnl_name = ch_list[chi];
      //read cont data
      std::string path = cont_name + "/" + chnl_name;
      try {
        support::TimingSys::restartEvent("readContinuous");
        cont_cfg = readContinuous(path, cfg, env.host_pCont);
        support::TimingSys::pauseEvent("readContinuous");
        //save the largest of all channels
        //note: the readContinuous chooses the smaller of
        // config file and the real npts.
        max_cont_npts = (max_cont_npts > cont_cfg.npts) ? 
            max_cont_npts : cont_cfg.npts;
      }
      catch (support::Exception e) {
        //logging
        support::LogSys::getLogger("userLog").warning(
            "Error read continuous file: " + path + "\n\n");
        continue;
      }
      //copy to device
      support::TimingSys::restartEvent("copyContinuous");
      cuda::memcpyH2D(env.dev_pCont, env.host_pCont, 
                      cont_cfg.npts * sizeof(float));
      support::TimingSys::pauseEvent("copyContinuous");
      //calculate mean and var
      support::TimingSys::restartEvent("calcContinuousMeanVar");
      getContMeanVar(env.dev_pCont, cont_cfg.npts, cfg.temp_npts(),
                     env.dev_pContMean, env.dev_pContVar);
      support::TimingSys::pauseEvent("calcContinuousMeanVar");

      //temp loop
      for (int ti = range.temp_start; ti < range.temp_end + 1; ++ti) {
        //clear temp data
        clearDevData(env.dev_pTemp, cfg.temp_npts());
        //clear cont data
        clearDevData(env.dev_pCorr, cfg.cont_npts());

        //test snr
        std::string temp_name = cfg.temp_list()[ti];
        std::string path = temp_name + "/" + cfg.snr_name();
        float snr;
        try {
          snr = readSNR(path, chnl_name);
        }
        catch (support::Exception e) {
          if (first_temp_loop) {
            support::LogSys::getLogger("userLog").warning(
                "Error read snr file: \n" + path + ". \nSet snr to 0\n\n");
          }
          snr = 0;
        }
        if (snr < cfg.snr_thr()) {
          if (first_temp_loop) {
            std::stringstream ss;
            ss << "template channel under threshold:\n";
            ss << temp_name + "/" + chnl_name + "\n";
            ss << "threshold: " + 
                support::Type2String<float>(cfg.snr_thr()) + '\n';
            ss << "value: " + support::Type2String<float>(snr) << "\n\n";
            support::LogSys::getLogger("userLog").warning(ss.str());
          }
          continue;
        }

        //read temp data
        path = temp_name + "/" + chnl_name;
        try {
          support::TimingSys::restartEvent("readTemplate");
          temp_cfg = readTemplate(path, cfg, env.host_pTemp);
          support::TimingSys::pauseEvent("readTemplate");
        }
        catch (support::Exception e) {
          //logging
          if (first_temp_loop) {
            support::LogSys::getLogger("userLog").warning(
                "Error read template file: " + path + "\n\n");
          }
          continue;
        }

        //calculate stack shift
        //if it is less than zero, discard.
        //int stack_shift = rint((temp_cfg.t - cfg.temp_tbefore() -
        //                       cont_cfg.b) / temp_cfg.delta);
        int stack_shift = rint((temp_cfg.t - cfg.temp_tbefore()) / temp_cfg.delta);
        if (stack_shift < 0) {
          std::stringstream ss;
          ss << "Stack shift less than zero: \n";
          ss << cont_name << "\n";
          ss << temp_name << "\n";
          support::LogSys::getLogger("userLog").warning(ss.str());
          continue;
        }

        //increase number of valid channel.
        valid_channels[ti - range.temp_start] ++;

        //calculate mean and var
        float mean, var;
        support::TimingSys::restartEvent("calcTemplateMeanVar");
        getTempMeanVar(env.host_pTemp, cfg.temp_npts(), mean, var);
        support::TimingSys::pauseEvent("calcTemplateMeanVar");
        //copy to device
        cuda::memcpyH2D(env.dev_pTemp, env.host_pTemp, 
                        cfg.temp_npts() * sizeof(float));
        //calculate correlation
        support::TimingSys::restartEvent("calcCorr");
        calcCorr(env.dev_pCorr, env.dev_pTemp, env.dev_pCont,
                 cont_cfg.npts, cfg.temp_npts(),
                 mean, var, env.dev_pContMean, env.dev_pContVar);
#if 0
        //dump correlation
        cuda::memcpyD2H(env.host_pCont, env.dev_pCorr, 
                        cfg.cont_npts() * sizeof(float));
        mkdir((cfg.out_root() + "/" + 
              support::splitString(temp_name, '/').back()).c_str(), 0777);
        std::string dump_path = cfg.out_root() + "/" + 
            support::splitString(temp_name, '/').back() + "/" + 
            support::splitString(cont_name, '/').back() + "_" + chnl_name;
        dumpResult(dump_path, env.host_pCont, cfg.cont_npts());
#endif 
        support::TimingSys::pauseEvent("calcCorr");
        //stack correlation
        //note: cont_cfg.npts has the smaller of config and real npts,
        //corr_size is cont_size - temp_size
        support::TimingSys::restartEvent("stackCorr");
        stack(env.dev_pCorr, 
              env.dev_pStack + (ti - range.temp_start) * cfg.cont_npts(),
              cont_cfg.npts - cfg.temp_npts(), stack_shift);
        support::TimingSys::pauseEvent("stackCorr");
      }
    }

#if 0
    for (int ti = range.temp_start; ti < range.temp_end + 1; ++ti) {
      //dump stack
      std::string temp_name = cfg.temp_list()[ti];
      cuda::memcpyD2H(env.host_pCont, 
                      env.dev_pStack + (ti - range.temp_start) * cfg.cont_npts(),
                      cfg.cont_npts() * sizeof(float));
      std::string dump_path = cfg.out_root() + "/" + 
          support::splitString(temp_name, '/').back() + "/" + 
          support::splitString(cont_name, '/').back() + "_stack";
      dumpResult(dump_path, env.host_pCont, cfg.cont_npts());
    }
#endif 

    //select
    for (int ti = range.temp_start; ti < range.temp_end + 1; ++ti) {
      std::string temp_name = cfg.temp_list()[ti];
      //check valid channels
      if (valid_channels[ti - range.temp_start] < cfg.num_chnlThr()) {
        //logging
        std::stringstream ss;
        ss << "Number of valid channel under threshold: \n";
        ss << cont_name << "\n";
        ss << temp_name << "\n";
        ss << "valid channel number: ";
        ss << support::Type2String<unsigned>(valid_channels[ti - range.temp_start]) + "\n";
        ss << "threshold: ";
        ss << support::Type2String<unsigned>(cfg.num_chnlThr()) + "\n\n";
        support::LogSys::getLogger("userLog").warning(ss.str());
        continue;
      }
      
      //copy back
      cuda::memcpyD2H(env.host_pCont, 
                      env.dev_pStack + 
                      (ti - range.temp_start) * cfg.cont_npts(),
                      cfg.cont_npts() * sizeof(float));
      //get MAD
      support::TimingSys::restartEvent("calcMAD");
      float* mad = getMAD(env.dev_pStack + 
                         (ti - range.temp_start) * cfg.cont_npts(), 
                         cfg.cont_npts());
      support::TimingSys::pauseEvent("calcMAD");

      //get path
      std::string cont_basename = support::splitString(cont_name, '/').back();
      std::string temp_basename = support::splitString(temp_name, '/').back();
      std::string path = cfg.out_root() + "/" + temp_basename;
      ///make the directory <out_root>/<temp_basename>
      mkdir(path.c_str(), 0777);
      ///get file path <out_root>/<temp_basename>/<mad_thr>times_<cont_basename>
      path += std::string("/") + 
          support::Type2String<float>(cfg.mad_ratio()) + 
          "times_" + cont_basename;
      std::ofstream ofs;
      ofs.open(path.c_str());
      user::throwError(ofs.is_open(), usererr::file_not_open, path);

      //select
      support::TimingSys::restartEvent("select");
      // the select size cont_npts is the largest of all channels
      // min(cfg.npts(), real npts)
      select(env.host_pCont, max_cont_npts, mad, cfg.mad_ratio(),
             cfg.sample_rate(), valid_channels[ti - range.temp_start], ofs);
      support::TimingSys::pauseEvent("select");
    }

    first_temp_loop = false;
  }
}
