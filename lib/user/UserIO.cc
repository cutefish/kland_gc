#include <cmath>
#include <cstdlib>
#include <vector>
#include <fstream>

#include "support/StringUtils.h"
#include "user/sacio.h"
#include "user/UserErrCategory.h"
#include "user/UserIO.h"

/* helper function, isSpecial() */
bool isSpecial(std::string path, Config cfg) {
  return path.find(cfg.special_channel()) != std::string::npos;
}

/* readTemplate() */
void readTemplate(std::string path, Config cfg, float* data) {
  SacInput temp_sac(path);
  float delta = temp_sac.header().delta;
  float event_time;
  if (isSpecial(path, cfg)) event_time = temp_sac.header().t1;
  else event_time = temp_sac.header().t2;
  float start_time = event_time - cfg.temp_tbefore();
  if (start_time < 0)
    user::throwError(0, usererr::temp_invalid_start_time, path);
  float end_time = event_time + cfg.temp_tafter();
  float init_time = temp_sac.header().b;
  size_t start_bytes = rint((start_time - init_time) / delta) * sizeof(float);
  size_t window_bytes = rint((end_time - start_time) / delta) * sizeof(float);

  temp_sac.read(reinterpret_cast<char*>(data), start_bytes, window_bytes);
}

/* readContinuous() */
size_t readContinuous(std::string path, Config cfg, float* data) {
  SacInput cont_sac(path);
  float delta = cont_sac.header().delta;
  float init_time = cont_sac.header().b;
  float final_time = cont_sac.header().e;
  size_t npts = cont_sac.header().npts;
  //choose the smaller 
  npts = (npts > cfg.cont_npts()) ? cfg.cont_npts() : npts;
  size_t bytes = npts * sizeof(float);
  cont_sac.read(reinterpret_cast<char*>(data), 0, bytes);
  return npts;
}

/* readSNR() */
float readSNR(std::string path, std::string channel) {
  std::ifstream ifs;
  std::string line;
  ifs.open(path.c_str());
  if (ifs.is_open()) {
    while (ifs.good()) {
      getline(ifs, line);
      if (line.find(channel) != std::string::npos) {
        std::vector<std::string> tokens = support::splitString(line, ' ');
        return atof(tokens[1].c_str());
      }
    }
  }
  else {
    user::throwError(0, usererr::file_not_open, path);
  }
  return 0;
}
