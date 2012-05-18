#include <cerrno>
#include <map>

#include "support/StdErrCategory.h"


namespace support {

namespace logging {

/*! \class Handler
 *  \brief Abstract class for output
 */
class Handler {
 public:
  virtual void handle(std::string message) = 0;
};

/*! \class StreamHandler
 *  \brief Stream handler for logger. It supports both std::cout and
 *  std::fstream object. It is a wrapper for the ostream objects.
 *
 * N.B.
 *  The user should promise that the lifetime of StreamHandler should 
 *  be longer than the ostream.
 */
class StreamHandler : public Handler {
 public:
  StreamHandler(std::ostream& out) : m_out(out) { }

  virtual void handle(std::string message) {
    m_out << message;
  }
 private:
  std::ostream& m_out;
};

/*! \class FileHandler
 *  \brief File handler for logger.
 */
class FileHandler : public Handler {
 public:
  FileHandler(const std::string file_name) {
    m_out.open(file_name);
    if (errno) {
      throw Exception(errno, getErrorCategory<StdErrCategory>(),
                      "LogFileHandler: " + file_name);
    }
  }

  ~FileHandler() {
    m_out.close();
  }

  virtual void handle(std::string message) {
    m_out << message;
  }
 private:
  std::ofstream m_out;
};

Logger& getLogger(const std::string name) {
  static std::map<std::string, Logger*> loggers;
  if (loggers.find(name) != loggers.end()) return *(loggers[name]);
  loggers[name] = new Logger();
}

void rmLogger(const std::string name) {
  if (loggers.find(name) == loggers.end()) return;
  Logger* to_delete = loggers[name];
  loggers
}

} /* namespace logging */

} /* namespace support */
