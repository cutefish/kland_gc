#include <cerrno>
#include <map>

#include "support/StdErrCategory.h"


namespace support {

namespace logging {

std::map<std::string, Logger*> getLoggerMap() {
  static std::map<std::string, Logger*> loggers;
  return loggers;
}

Logger& getLogger(const std::string name) {
  std::map<std::string, Logger*>& loggers = getLoggerMap();
  if (loggers.find(name) != loggers.end()) return *(loggers[name]);
  loggers[name] = new Logger();
}

void rmLogger(const std::string name) {
  std::map<std::string, Logger*>& loggers = getLoggerMap();
  if (loggers.find(name) == loggers.end()) return;
  Logger* to_delete = loggers[name];
  loggers.
}

} /* namespace logging */

} /* namespace support */
