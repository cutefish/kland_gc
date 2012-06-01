#include <iostream>

#include "Config.h"
#include "Hosts.h"
#include "support/Logging.h"

static void usage() {
  std::cerr<<"Usage: kland_gc <config_path> <log_root> <out_root>"<<endl;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    usage();
    exit(1);
  }
  Config cfg(argv[1], argv[2], argv[3]);

  RunEnv env = init(cfg);

  support::LogSys::getLogger("userLog").info(cfg.repr());

  if (env.rank == 0) {
    doLeaderLoop(env, cfg);
  }
  else {
    doWorkerLoop(env, cfg);
  }

  finalize(env);

}
