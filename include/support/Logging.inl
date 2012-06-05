#include <iostream>
#include <stdexcept>

#include "StdErrCategory.h"
#include "Type.h"

namespace support {

/*** Logger ***/
/* ctor/dtor */
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
inline void Logger::debug(const std::string& message) {
  log(DEBUG, "[" + m_name + ":DEBUG]" + message);
}

/* Logger::info() */
inline void Logger::info(const std::string& message) {
  log(INFO, "[" + m_name + ":INFO]" + message);
}

/* Logger::warning() */
inline void Logger::warning(const std::string& message) {
  log(WARNING, "[" + m_name + ":WARNING]" + message);
}

/* Logger::error() */
inline void Logger::error(const std::string& message) {
  log(ERROR, "[" + m_name + ":ERROR]" + message);
}

/* Logger::critical() */
inline void Logger::critical(const std::string& message) {
  log(CRITICAL, "[" + m_name + ":CRITICAL]" + message);
}

/* Logger::log() */
inline void Logger::log(const Level lvl, const std::string& message) {
  if (!isEnabledFor(lvl)) return;
  for (std::map<std::string, HandlerRef>::iterator it = m_handlers.begin();
       it != m_handlers.end(); ++it) {
    if (((*it).second)->isEnabledFor(lvl)) {
      ((*it).second)->emit(message);
    }
  }
}

/* Logger::addHandler() */
inline void Logger::addHandler(const HandlerRef& hdlr, 
                               const std::string& name) {
  if (name == "") {
    m_handlers.insert(std::pair<std::string, HandlerRef>(
            Type2String<int>(m_handlers.size()), hdlr));
  }
  else {
    m_handlers.insert(std::pair<std::string, HandlerRef>(name, hdlr));
  }
}

/* Logger::removeHandler() */
inline void Logger::removeHandler(const std::string& name) {
  m_handlers.erase(name);
}

/*** LogSys ***/
/* init() */
inline void LogSys::init() {
  getSet();
}

/* finalize() */
inline void LogSys::finalize() {
  for (std::map<std::string, Logger*>::iterator it = getSet()->begin();
       it != getSet()->end(); ++it) {
    delete (*it).second;
  }
}

/* newLogger() */
inline Logger& LogSys::newLogger(const std::string& name) {
  if (getSet()->find(name) != getSet()->end())
    return *((*getSet())[name]);
  (*getSet())[name] = new Logger(name);
  return *((*getSet())[name]);
}

/* delLogger() */
inline void LogSys::delLogger(const std::string& name) {
  getSet()->erase(name);
}

/* getLogger() */
inline Logger& LogSys::getLogger(const std::string& name) {
  checkNameInTableOrDie(name);
  return *((*getSet())[name]);
}

/* getSet()
 * get a static set pointer
 */
inline std::map<std::string, Logger*>* LogSys::getSet() {
  static std::map<std::string, Logger*> ret;
  return &ret;
}

/* checkNameInTableOrDie() */
inline void LogSys::checkNameInTableOrDie(const std::string& name){
  if (getSet()->find(name) == getSet()->end()) {
    throw std::runtime_error(
        "LogSys logger name not in table: " + name);
  }
}

} /* namespace support */
