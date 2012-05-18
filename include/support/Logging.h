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

namespace support {

namespace logging {

enum Level {
  DEBUG,
  INFO,
  WARNING,
  ERROR,
  CRITICAL
};

class Handler;
class StreamHandler;
class FileHandler;

} /* namespace logging */

Logger& getLogger(const std::string name);

void rmLogger(const std::string name);

void rmAllLoggers();

/*! \class Logger
 *  \brief Objects of Logger class support five logging levels: DEBUG, INFO,
 *  WARNING, ERROR, CRITICAL. The default level is WARNING. Default logging
 *  outputs messages to standard error, however, output to file handler can be
 *  set. 
 *  
 *  To Do
 *    Hierarchical structure loggers, that is, messages from the child
 *  can be propagated to the parents.
 *    Filter of logs
 */
class Logger {
 public:
  /*** ctor/dtor ***/
  //no ctors. New loggers are created through gettor
  friend Logger& getLogger(const std::string name);
  friend void rmLogger(const std::string name);
  
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
  bool isEnabledFor(const Level lvl);

  /* getEffectiveLevel() */
  Level getEffectiveLevel();

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

  /* log() */
  void log(const Level lvl, const std::string& message);

  /* addFilter() */
  //void addFilter(const Filter& );

  /* removeFilter() */
  //void removeFilter()

  /* addHandler() */
  void addHandler(const Handler& hdlr);

  /* removeHandler() */
  void removeHandler(const Handler& hdlr);

 private:
  /* ctors/dtor */
  Logger();
  Logger(const Logger& other);
  Logger& operator=(const Logger& other);
  ~Logger();


};

} /* namespace support */

#endif /* SUPPORT_LOGGING_H_ */
