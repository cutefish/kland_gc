namespace support
{

inline Exception ::Exception(ErrorCode ec, const std::string &what_arg)
    : std::runtime_error(what_arg), m_error_code(ec) {
} // end Exception::Exception()


inline Exception ::Exception(ErrorCode ec, const char *what_arg)
    : std::runtime_error(what_arg), m_error_code(ec) {
} // end Exception::Exception()


inline Exception ::Exception(ErrorCode ec)
    : std::runtime_error(""), m_error_code(ec) {
} // end Exception::Exception()


inline Exception
  ::Exception(int ev, const ErrorCategory &ecat, const std::string &what_arg)
    : std::runtime_error(what_arg), m_error_code(ev, ecat) {
} // end Exception::Exception()


inline Exception
  ::Exception(int ev, const ErrorCategory &ecat, const char *what_arg)
    : std::runtime_error(what_arg), m_error_code(ev, ecat) {
} // end Exception::Exception()


inline Exception
  ::Exception(int ev, const ErrorCategory &ecat)
    : std::runtime_error(""), m_error_code(ev, ecat) {
} // end Exception::Exception()


inline const ErrorCode& Exception ::code(void) const throw() {
  return m_error_code;
} // end Exception::code()


inline const char *Exception ::what(void) const throw() {
  if(m_what.empty())
  {
    try
    {
      m_what = this->std::runtime_error::what();
      if(m_error_code)
      {
        if(!m_what.empty()) m_what += ": ";
        m_what += m_error_code.message();
      }
    }
    catch(...)
    {
      return std::runtime_error::what();
    }
  }

  return m_what.c_str();
} // end Exception::what()

} // end system
