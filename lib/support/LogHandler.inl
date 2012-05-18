namespace support {

namespace logging {

StreamHandler::StreamHandler(std::ostream& out) : m_out(out) { }

virtual void StreamHandler::handle(std::string message) {
  m_out << message;
}

FileHandler::FileHandler(const std::string file_name) {
  m_out.open(file_name);
  if (errno) {
    throw Exception(errno, getErrorCategory<StdErrCategory>(),
                    "LogFileHandler: " + file_name);
  }
}

FileHandler::~FileHandler() {
  m_out.close();
}

virtual void FileHandler::handle(std::string message) {
  m_out << message;
}

} /* namespace logging */

} /* namespace support */
