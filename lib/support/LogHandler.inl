#include <cerrno>
#include <fstream>
#include <iostream>

#include "support/StdErrCategory.h"
#include "support/Exception.h"

namespace support {

namespace logging {

/*** Handler ***/
inline Handler::Handler() : m_lvl(DEBUG), m_out(NULL) { }

inline Handler::~Handler() { }

inline std::ostream& Handler::get() { return *m_out; }

inline void Handler::set(std::ostream& out) { m_out = &out; }

inline void Handler::emit(std::string message) {
  if ((m_out) && (m_out->good())) {
    (*m_out) << message;
    m_out->flush();
  }
}

inline void Handler::setLevel(Level lvl) {
  m_lvl = lvl;
}

inline Level Handler::getLevel() const{
  return m_lvl;
}

inline bool Handler::isEnabledFor(Level lvl) const {
  return lvl >= m_lvl;
}

/*** StreamHandler ***/
inline StreamHandler& StreamHandler::create(std::ostream& out) {
  StreamHandler* ret = new StreamHandler(out);
  return (*ret);
}

inline void StreamHandler::destroy(StreamHandler& hdlr) {
  delete &hdlr;
}

inline StreamHandler::StreamHandler(std::ostream& out) { 
  set(out);
}

inline StreamHandler::~StreamHandler() {
}


/*** FileHandler ***/
inline FileHandler& FileHandler::create(std::string name) {
  FileHandler* ret = new FileHandler(name);
  return (*ret);
}

inline void FileHandler::destroy(FileHandler& hdlr) {
  delete &hdlr;
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

inline FileHandler::~FileHandler() {
  std::ofstream* f_ = dynamic_cast<std::ofstream*> (&get());
  std::cout << "closing handler\n";
  f_->close();
  delete f_;
}

inline std::string FileHandler::name() const {
  return m_name;
}

} /* namespace logging */

} /* namespace support */
