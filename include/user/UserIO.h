#ifndef USER_USERIO_H_
#define USER_USERIO_H_

#include <string>

#include "user/Config.h"

struct TempConfig {
};

/* readTemplate()
 */
void readTemplate(std::string path, Config cfg, float* data);

/* readContinuous()
 */
void readContinuous(std::string path, Config cfg, float* data);

/* readSNR()
 */
float readSNR(std::string path, std::string channel);

#endif /* USER_USERIO_H_ */
