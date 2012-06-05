#include <cerrno>
#include <fstream>
#include <iostream>

#include "StdErrCategory.h"
#include "Exception.h"

namespace support {

/*** HandlerRef ***/
/* ctor/dtor */
inline HandlerRef::HandlerRef(Handler* handler_ptr) : m_ptr(handler_ptr) {
  m_ptr->incRef();
}

inline HandlerRef::HandlerRef(const HandlerRef& ref) : m_ptr(ref.m_ptr) {
  m_ptr->incRef();
}

inline HandlerRef& HandlerRef::operator= (const HandlerRef& ref) {
  //gracefully handle self assignment
  if (m_ptr == ref.m_ptr) return (*this);
  //decRef the current referece
  unsigned num_refs = m_ptr->decRef();
  if (num_refs == 0) { delete m_ptr; }
  m_ptr = ref.m_ptr;
  m_ptr->incRef();
}

inline HandlerRef::~HandlerRef() {
  unsigned num_refs = m_ptr->decRef();
  if (num_refs == 0) { delete m_ptr; }
}

inline void HandlerRef::emit(std::string message) { m_ptr->emit(message); }

inline void HandlerRef::setLevel(Level lvl) { m_ptr->setLevel(lvl); }

inline Level HandlerRef::getLevel() const { return m_ptr->getLevel(); }

inline bool HandlerRef::isEnabledFor(Level lvl) const { 
  return m_ptr->isEnabledFor(lvl); 
}

inline Handler* HandlerRef::operator->() { return m_ptr; }

/*** Handler ***/

inline Handler::Handler() : 
    m_lvl(WARNING), m_out(NULL), m_mtx(), m_refCount(0) { }

inline Handler::~Handler() { }

/* Handler::get() */
inline std::ostream& Handler::get() { return *m_out; }

/* Handler::set() */
inline void Handler::set(std::ostream& out) { m_out = &out; }

/* Handler::emit() */
inline void Handler::emit(std::string message) {
  if ((m_out) && (m_out->good())) {
    //using lock to make thread safety
    Lock lock(m_mtx);
    (*m_out) << message;
    //unlocked automatically;
  }
}

/* Handler::setLevel() */
inline void Handler::setLevel(Level lvl) {
  m_lvl = lvl;
}

/* Handler::getLevel() */
inline Level Handler::getLevel() const{
  return m_lvl;
}

/* Handler::isEnabledFor() */
inline bool Handler::isEnabledFor(Level lvl) const {
  return lvl >= m_lvl;
}

/* incRef() */
inline unsigned Handler::incRef() {
  Lock lock(m_mtx);
  m_refCount ++;
  return m_refCount;
}

/* decRef() */
inline unsigned Handler::decRef() {
  Lock lock(m_mtx);
  m_refCount --;
  return m_refCount;
}

/*** StreamHandler ***/
/* create() */
inline HandlerRef StreamHandler::create(std::ostream& out) {
  StreamHandler* ptr = new StreamHandler(out);
  return HandlerRef(ptr);
}

/* StreamHandler::~StreamHandler() */
inline StreamHandler::~StreamHandler() { }

inline StreamHandler::StreamHandler(std::ostream& out) { 
  set(out);
}

/*** FileHandler ***/

inline HandlerRef FileHandler::create(const std::string file_name) {
  FileHandler* ptr = new FileHandler(file_name);
  return HandlerRef(ptr);
}

inline FileHandler::~FileHandler() {
  std::ofstream* f_ = dynamic_cast<std::ofstream*> (&get());
  f_->close();
  delete f_;
}

inline FileHandler::FileHandler(const std::string file_name) 
  : m_name(file_name) 
{
  std::ofstream* f_ = new std::ofstream;
  f_->open(file_name.c_str());
  if (errno) {
    throw Exception(errno, getErrorCategory<StdErrCategory>(),
                    "LogFileHandler: " + file_name);
  }
  set(*f_);
}

inline std::string FileHandler::name() const {
  return m_name;
}

} /* namespace support */
