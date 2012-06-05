#ifndef SUPPORT_LOGGING_H_
#define SUPPORT_LOGGING_H_

/*! \file Logging.h
 *  \brief A logging utility. 
 *
 *  This utility is designed for single threaded usage, that is, not thread-safe
 *
 *  To Do: thread-safety
 *
 */

#include <string>
#include <map>

#include "LogHandler.h"

namespace support {

/*! \class Logger
 *  \brief Objects of Logger class support five logging levels: DEBUG, INFO,
 *  WARNING, ERROR, CRITICAL. Loggings with level under the setting will be
 *  ignored.
 *  
 *  To Do
 *    Hierarchical structure loggers, that is, messages from the child
 *  can be propagated to the parents.
 *    Filter of logs
 */
class Logger {
 public:
  /*** ctor/dtor ***/
  Logger(std::string name);

  /*** methods ***/

  /* propagate()
   * if return True, messages of itself and its child are propagate to the
   * parents */
  //bool propagate();

  /* setLevel()
   * set the level of the logger, messages of levels equal or higher will be
   * printed. */
  void setLevel(const Level lvl);

  /* isEnabledFor() */
  bool isEnabledFor(const Level lvl) const;

  /* getEffectiveLevel() */
  Level getEffectiveLevel() const;

  /* getChild() */
  //Logger getChild(const std::string& name);

  /* debug() */
  void debug(const std::string& message);

  /* info() */
  void info(const std::string& message);

  /* warning() */
  void warning(const std::string& message);

  /* error() */
  void error(const std::string& message);

  /* critical() */
  void critical(const std::string& message);

  /* log() */
  void log(const Level lvl, const std::string& message);

  /* addFilter() */
  //void addFilter(const Filter& );

  /* removeFilter() */
  //void removeFilter()

  /* addHandler() */
  void addHandler(const HandlerRef& hdlr, const std::string& name="");

  /* removeHandler() */
  void removeHandler(const std::string& name);

 private:
  /* ctors/dtor */
  Logger(const Logger& other);
  Logger& operator=(const Logger& other);

  /* member variable */
  std::string m_name;
  Level m_lvl;
  std::map<std::string, HandlerRef> m_handlers;
};

/*!\class LogSys
 * \brief static interface for easier access to loggers and handlers
 */
class LogSys {
 public:
  /*** ctor/dtor ***/
  static void init();
  static void finalize();

  /*** member function ***/
  /* newLogger()
   * insert a new logger
   */
  static Logger& newLogger(const std::string& name);

  /* delLogger()
   * remove a logger
   */
  static void delLogger(const std::string& name);

  /* getLogger()
   * get the logger reference, raise error if name not found.
   */
  static Logger& getLogger(const std::string& name);

 private:
  static std::map<std::string, Logger*>* getSet();
  static void checkNameInTableOrDie(const std::string& name);
};

} /* namespace support */

#include "Logging.inl"

#endif /* SUPPORT_LOGGING_H_ */
