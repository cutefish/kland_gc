#include <map>

#include <iostream>

#include "support/StdErrCategory.h"

namespace support {

namespace logging {

/* getLoggerMap() */
inline std::map<std::string, Logger*>& getLoggerMap() {
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
inline Logger& getLogger(const std::string name) {
  std::map<std::string, Logger*>& loggers = getLoggerMap();
  if (loggers.find(name) != loggers.end()) return *(loggers[name]);
  loggers[name] = new Logger(name);
  return *(loggers[name]);
}

/* ctors */
inline Logger::Logger(std::string name) : m_name(name), m_lvl(WARNING) { } 

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
  log(DEBUG, "[DEBUG]" + message);
}

inline void Logger::info(const std::string& message) const {
  log(INFO, "[INFO]" + message);
}

inline void Logger::warning(const std::string& message) const {
  log(WARNING, "[WARNING]" + message);
}

inline void Logger::error(const std::string& message) const {
  log(ERROR, "[ERROR]" + message);
}

inline void Logger::critical(const std::string& message) const {
  log(CRITICAL, "[CRITICAL]" + message);
}

inline void Logger::log(const Level lvl, const std::string& message) const {
  if (!isEnabledFor(lvl)) return;
  for (std::list<Handler*>::const_iterator it = m_handlers.begin();
       it != m_handlers.end(); ++it) {
    if ((*it)->isEnabledFor(lvl)) {
      (*it)->emit(message);
    }
  }
}

inline void Logger::addHandler(Handler& hdlr) {
  m_handlers.push_back(&hdlr);
}

inline void Logger::removeHandler(Handler& hdlr) {
  m_handlers.remove(&hdlr);
}

inline Logger::~Logger() {
  //this is the end of the program
  //we should destroy the handlers
  for (std::list<Handler*>::iterator it = m_handlers.begin();
       it != m_handlers.end(); ++it) {
    if ((*it) != NULL) {
      std::cout << "delete handler\n";
      delete (*it);
    }
  }
}

} /* namespace logging */

} /* namespace support */
