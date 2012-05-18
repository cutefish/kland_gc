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
#include <vector>

namespace support {

namespace logging {

enum Level {
  DEBUG = 0,
  INFO = 1,
  WARNING = 2,
  ERROR = 3,
  CRITICAL = 4
};

class Handler;
class StreamHandler;
class FileHandler;

/* getLogger()
 * Return a Logger instance.
 * Default level: WARNING
 * Default handler: std::cout
 * Multiple calls with the same name returns the same instance.
 */
Logger& getLogger(const std::string name);

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
  //no ctors. New loggers are created through gettor
  friend Logger& getLogger(const std::string name);
  
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
  void debug(const std::string& message) const;

  /* info() */
  void info(const std::string& message) const;

  /* warning() */
  void warning(const std::string& message) const;

  /* error() */
  void error(const std::string& message) const;

  /* critical() */
  void critical(const std::string& message) const;

  /* log() */
  void log(const Level lvl, const std::string& message) const;

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

  /* member variable */
  Level m_lvl;
  std::vector<Handler&> m_handlers;
};

} /* namespace logging */


} /* namespace support */

#endif /* SUPPORT_LOGGING_H_ */
