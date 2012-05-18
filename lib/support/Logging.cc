#include <cerrno>
#include <map>

#include "support/StdErrCategory.h"


namespace support {

namespace logging {

/* getLoggerMap() */
std::map<std::string, Logger*> getLoggerMap() {
  static std::map<std::string, Logger*> loggers;
  return loggers;
}

/* getLogger()
 *
 * N.B.
 * The loggers are never deleted until the program ends.
 * Considering there will be only a limited number of loggers
 * this memory leak will not cause serious trouble.
 */
Logger& getLogger(const std::string name) {
  std::map<std::string, Logger*>& loggers = getLoggerMap();
  if (loggers.find(name) != loggers.end()) return *(loggers[name]);
  loggers[name] = new Logger();
}

}

/* setLevel() */
inline void Logger::setLevel(const Level lvl) {
  m_lvl = lvl;
}

/* isEnabledFor() */
inline bool Logger::isEnabledFor(const Level lvl) const {
  return lvl >= m_lvl;
}

inline Level Logger::getEffectiveLevel() const {
  return m_lvl;
}

inline void Logger::debug(const std::string& message) const {
  log(DEBUG, message);
}

inline void Logger::info(const std::string& message) const {
  log(INFO, message);
}

inline void Logger::warning(const std::string& message) const {
  log(WARNING, message);
}

inline void Logger::error(const std::string& message) const {
  log(ERROR, message);
}

inline void Logger::critical(const std::string& message) const {
  log(CRITICAL, message);
}

inline void Logger::log(const Level lvl, const std::string& message) const {
  if (!isEnabledFor(lvl)) return;
  for (int i = 0; i < m_handlers.size(); ++i) {
    if (m_handlers[i].isEnabledFor(lvl)) {
      m_handlers[i].emit(message);
    }
  }
}

inline void addHandler(const Handler& hdlr) {
  m_handlers.push_back(hdlr);
}

inline void removeHandler(const Handler& hdlr) {
  m_handlers.erase(hdlr);
}

inline Logger::Logger() : m_lvl(WARNING) { } 

} /* namespace logging */

} /* namespace support */
