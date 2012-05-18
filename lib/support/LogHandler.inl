#include <cerrno>

#include "support/StdErrCategory.h"

namespace support {

namespace logging {

inline StreamHandler::StreamHandler(std::ostream& out)
    : m_out(out) {
}

virtual inline void StreamHandler::handle(std::string message) {
  m_out << message;
}

inline FileHandler::FileHandler(std::string file_name) {
}

inline FileHandler::~FileHandler() {
  m_out.close();
}

virtual inline void FileHandler::handle(std::string message) {
  m_out << message;
}

} /* namespace logging */

} /* namespace support */
