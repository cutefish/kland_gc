#ifndef SUPPORT_LOGHANDLER_H_
#define SUPPORT_LOGHANDLER_H_

namespace support {

namespace logging {

enum Level {
  DEBUG = 0,
  INFO = 1,
  WARNING = 2,
  ERROR = 3,
  CRITICAL = 4
};


/*! \class Handler
 *  \brief Abstract class for output
 */
class Handler {
 public:

  /*** ctor/dtor ***/
  Handler();

  virtual ~Handler();

  /*** gettor/settor ***/
  std::ostream& get();
  void set(std::ostream& out);

  /*** member function ***/
  /* emit()
   * Emit a message.
   * It is possible that the underlying stream is broken, in this case
   * emit() has no effect
   */
  virtual void emit(std::string message);

  /* setLevel()
   * Set the emit level. 
   */
  void setLevel(Level lvl);

  /* getLevel() */
  Level getLevel() const;

  /* isEnabledFor() */
  bool isEnabledFor(Level lvl) const;

 private:
  Level m_lvl;
  std::ostream* m_out;
};

/*! \class StreamHandler
 *  \brief Stream handler for logger. It supports both std::cout and
 *  std::fstream object. It is a wrapper for the ostream objects.
 *
 *  The effective lifetime of the StreamHandler is the same as the ostream
 *  object itself. this->emit() after the ostream object destruction will have
 *  no effect.
 *
 */

class StreamHandler : public Handler {
 public:
  /*** ctor/dtor ***/
  /*create()
   * ensure the handler lives long
   */
  static StreamHandler& create(std::ostream& out);

  static void destroy(StreamHandler& hdlr);

  virtual ~StreamHandler();

 private:
  /*** private ctor ***/
  StreamHandler(std::ostream& out);
  StreamHandler(const StreamHandler& other);
  StreamHandler& operator=(const StreamHandler& other);

};

/*! \class FileHandler
 *  \brief File handler for logger.
 */
class FileHandler : public Handler {
 public:
  /*** ctor/dtor ***/
  /* create()
   * ensure the handler lives long
   */
  static FileHandler& create(std::string name);

  /* destroy() */
  static void destroy(FileHandler& hdlr);

  virtual ~FileHandler();

  /*** getter ***/
  std::string name() const;
 private:
  /*** private ctor ***/
  FileHandler(const std::string file_name);
  FileHandler(const FileHandler& other);
  FileHandler& operator=(const FileHandler& other);

  std::string m_name;
};

} /* namespace logging */

} /* namespace support */

#include "../lib/support/LogHandler.inl"

#endif /* SUPPORT_LOGHANDLER_H_ */
