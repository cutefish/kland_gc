#ifndef USER_USERIO_H_
#define USER_USERIO_H_

#include <string>

#include "user/Config.h"

struct DataConfig {
  float delta;
  float b;
  float t;
  size_t npts;
};

/* readTemplate()
 */
DataConfig readTemplate(std::string path, Config cfg, float* data);

/* readContinuous()
 * Read continuous and return actual size.
 */
DataConfig readContinuous(std::string path, Config cfg, float* data);

/* readSNR()
 */
float readSNR(std::string path, std::string channel);

#endif /* USER_USERIO_H_ */
