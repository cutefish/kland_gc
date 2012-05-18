namespace support {

namespace logging {

inline Handler::Handler() : m_lvl(WARNING) { }

inline void Handler::setLevel(Level lvl) {
  m_lvl = lvl;
}

inline Level Handler::getLevel() const{
  return m_lvl;
}

inline bool Handler::isEnabledFor(Level lvl) const {
  return lvl >= m_lvl;
}

inline StreamHandler::StreamHandler(std::ostream& out) : m_out(out) { }

inline virtual void StreamHandler::emit(std::string message) {
  m_out << message;
}

inline FileHandler::FileHandler(const std::string file_name) {
  m_out.open(file_name);
  if (errno) {
    throw Exception(errno, getErrorCategory<StdErrCategory>(),
                    "LogFileHandler: " + file_name);
  }
}

inline FileHandler::~FileHandler() {
  m_out.close();
}

inline virtual void FileHandler::emit(std::string message) {
  m_out << message;
}

} /* namespace logging */

} /* namespace support */
