#ifndef SUPPORT_LOGHANDLER_H_
#define SUPPORT_LOGHANDLER_H_

namespace logging {

/*! \file LogHandler.h
 *  \brief Output handling for logging. 
 *
 */

/*! \class Handler
 *  \brief Abstract class for output
 */
class Handler {
  virtual void handle() = 0;
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
  StreamHandler(const std::ostream& out);
privae:
  std::ostream& _out;
};

/*! \class FileHandler
 *  \brief File handler for logger.
 */
class FileHandler : public Handler {
 public:
  FileHandler(const std::string file_name);
  ~FileHandler();
 private:
  std::ofstream _out;
};

} /* namespace logging */

#endif /* SUPPORT_LOGHANDLER_H_ */
