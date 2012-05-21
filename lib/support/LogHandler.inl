#include <cerrno>
#include <fstream>
#include <iostream>

#include "support/StdErrCategory.h"
#include "support/Exception.h"

namespace logging {


/*** Handler ***/
/* Handler::Handler() */
inline Handler::Handler() : m_lvl(DEBUG), m_out(NULL) { }

/* Handler::~Handler() */
inline Handler::~Handler() { }

/* Handler::get() */
inline std::ostream& Handler::get() { return *m_out; }

/* Handler::set() */
inline void Handler::set(std::ostream& out) { m_out = &out; }

/* Handler::emit() */
inline void Handler::emit(std::string message) {
  if ((m_out) && (m_out->good())) {
    //using lock to make thread safety
    support::Lock lock(m_mtx);
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

/*** StreamHandler ***/
/* StreamHandler::StreamHandler() */
inline StreamHandler::StreamHandler(std::ostream& out) { 
  set(out);
}

/* StreamHandler::~StreamHandler() */
inline StreamHandler::~StreamHandler() {
}


/*** FileHandler ***/
inline FileHandler::FileHandler(const std::string file_name) 
  : m_name(file_name) 
{
  std::ofstream* f_ = new std::ofstream;
  f_->open(file_name.c_str());
  if (errno) {
    throw support::Exception(errno, support::getErrorCategory<support::StdErrCategory>(),
                    "LogFileHandler: " + file_name);
  }
  set(*f_);
}

inline FileHandler::~FileHandler() {
  std::ofstream* f_ = dynamic_cast<std::ofstream*> (&get());
  f_->close();
  delete f_;
}

inline std::string FileHandler::name() const {
  return m_name;
}

} /* namespace logging */
