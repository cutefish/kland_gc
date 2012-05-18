#ifndef SUPPORT_LOGHANDLER_H_
#define SUPPORT_LOGHANDLER_H_

#include "support/Logging.h"

namespace support {

namespace logging {

/*! \class Handler
 *  \brief Abstract class for output
 */
class Handler {
 public:
  Handler();

  virtual void emit(std::string message) = 0;

  void setLevel(Level lvl);

  Level getLevel(Level lvl) const;

  bool isEnabledFor(Level lvl) const;

 private:
  Level m_lvl;
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
  StreamHandler(std::ostream& out);

  virtual void emit(std::string message);

 private:
  std::ostream& m_out;
};

/*! \class FileHandler
 *  \brief File handler for logger.
 */
class FileHandler : public Handler {
 public:
  FileHandler(const std::string file_name);

  ~FileHandler();

  virtual void emit(std::string message);
 private:
  std::ofstream m_out;
};

} /* namespace logging */

} /* namespace support */

#include "../lib/support/LogHandler.inl"

#endif /* SUPPORT_LOGHANDLER_H_ */
