#ifndef SUPPORT_LOGHANDLER_H_
#define SUPPORT_LOGHANDLER_H_

#include "ThreadSynch.h"

namespace support {

enum Level {
  DEBUG = 0,
  INFO = 1,
  WARNING = 2,
  ERROR = 3,
  CRITICAL = 4
};

class Handler;

/*!\class HandlerRef
 * \brief Handler reference class. The handlers are passing around by this class
 * objects. The objects inc or dec the reference count of Handlers. The object
 * is responsible for destroy the handler if there is no other one refer to it.
 */
class HandlerRef {
 public:
  /*** ctor/dtor ***/
  HandlerRef(Handler* handler_ptr);
  HandlerRef(const HandlerRef& handler_ref);
  HandlerRef& operator= (const HandlerRef& handler_ref);
  ~HandlerRef();

  /*** member function ***/
  /* wrapper */
  void emit(std::string message);
  void setLevel(Level lvl);
  Level getLevel() const;
  bool isEnabledFor(Level lvl) const;

  /* getter */
  /* operator ->()
   * This accessor might be confusing for a Reference class, however, we cannot
   * overload dot operator.
   */
  Handler* operator->();

 private:
  Handler* m_ptr;
};


/*! \class Handler
 *  \brief Abstract class for output. The objects of this class only lives on
 *  heap, that is, there is no public constructor. Objects are created using the
 *  create() method and can only be accessed by HandlerRef objects. The object
 *  keeps track of the references count, so that it can be destroyed
 *  automatically.
 */
class Handler {
 public:

  /*** ctor/dtor ***/
  //static HandlerRef create();

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

  /* incRef()
   * Increase the reference count and return;
   */
  unsigned incRef();

  /* decRef()
   * Decrease the reference count and return;
   */
  unsigned decRef();

 protected:
  /*** protected ctor for inheritance ***/
  Handler();

 private:
  Level m_lvl;
  std::ostream* m_out;
  Mutex m_mtx;
  unsigned m_refCount;
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
  static HandlerRef create(std::ostream& out);

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
  static HandlerRef create(const std::string file_name);

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

} /* namespace support */

#include "LogHandler.inl"

#endif /* SUPPORT_LOGHANDLER_H_ */
