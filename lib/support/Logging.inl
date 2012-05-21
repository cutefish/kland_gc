#include <map>

#include <iostream>

#include "support/StdErrCategory.h"

namespace logging {

/*! \class LoggingMap
 *  \brief The class is the set of loggers of the program. To give each loggers
 *  a lifetime equal to the program, we have to destroy the loggers during
 *  program termination, which is done by the destructor of this LoggingMap
 *  class.
 */
class LoggingMap {
 public:
  /*** ctor/dtor ***/
  //map destructor delete all loggers on heap
  ~LoggingMap() {
    for (std::map<std::string, Logger*>::iterator it = m_map.begin();
         it != m_map.end(); ++it) {
      delete (*it).second;
    }
  }

  /*** gettor/settor ***/
  std::map<std::string, Logger*>& map() { return m_map; }
 private:
  std::map<std::string, Logger*> m_map;
};

/* getLogger()
 */
inline Logger& getLogger(const std::string name) {
  static LoggingMap the_map;
  std::map<std::string, Logger*>& loggers = the_map.map();
  if (loggers.find(name) != loggers.end()) return *(loggers[name]);
  loggers[name] = new Logger(name);
  return *(loggers[name]);
}

/* Logger::ctors */
inline Logger::Logger(std::string name) : m_name(name), m_lvl(WARNING) { } 

/* Logger::setLevel() */
inline void Logger::setLevel(const Level lvl) {
  m_lvl = lvl;
}

/* Logger::isEnabledFor() */
inline bool Logger::isEnabledFor(const Level lvl) const {
  return lvl >= m_lvl;
}

/* Logger::getEffectiveLevel() */
inline Level Logger::getEffectiveLevel() const {
  return m_lvl;
}

/* Logger::debug() */
inline void Logger::debug(const std::string& message) const {
  log(DEBUG, "[DEBUG]" + message);
}

/* Logger::info() */
inline void Logger::info(const std::string& message) const {
  log(INFO, "[INFO]" + message);
}

/* Logger::warning() */
inline void Logger::warning(const std::string& message) const {
  log(WARNING, "[WARNING]" + message);
}

/* Logger::error() */
inline void Logger::error(const std::string& message) const {
  log(ERROR, "[ERROR]" + message);
}

/* Logger::critical() */
inline void Logger::critical(const std::string& message) const {
  log(CRITICAL, "[CRITICAL]" + message);
}

/* Logger::log() */
inline void Logger::log(const Level lvl, const std::string& message) const {
  if (!isEnabledFor(lvl)) return;
  for (std::list<Handler*>::const_iterator it = m_handlers.begin();
       it != m_handlers.end(); ++it) {
    if ((*it)->isEnabledFor(lvl)) {
      (*it)->emit(message);
    }
  }
}

/* Logger::addHandler() */
inline void Logger::addHandler(Handler& hdlr) {
  m_handlers.push_back(&hdlr);
}

/* Logger::removeHandler() */
inline void Logger::removeHandler(Handler& hdlr) {
  m_handlers.remove(&hdlr);
}

inline Logger::~Logger() {
}

} /* namespace logging */
