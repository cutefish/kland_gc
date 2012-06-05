namespace support {

/* ErrorCategory impl */

inline ErrorCategory::~ErrorCategory() { }

inline ErrorCode ErrorCategory::default_code(int ev) const {
  return ErrorCode(ev, *this);
} // end ErrorCategory::default_code() 

inline bool ErrorCategory::operator==(const ErrorCategory& rhs) const {
  return this == &rhs;
} // end ErrorCategory::operator==()

inline bool ErrorCategory::operator!=(const ErrorCategory& rhs) const {
  return !this->operator==(rhs);
} // end ErroCategory::operator!=()

inline bool ErrorCategory::operator<(const ErrorCategory& rhs) const {
  return (this < &rhs);
} // end operator<()

/* only instance get method */
template<typename Category>
const Category& getErrorCategory() {
  static const Category cat;
  return cat;
}

/* UnknownErrorCategory */

class UnknownErrorCategory : public ErrorCategory {
 public:
  typedef enum UnknownError {
    Unknown
  } type;

  UnknownErrorCategory() { }

  inline virtual const char* name() const {
    return "UnknownErrorCategory";
  }

  template<typename T> void ignore(const T&) const { }

  inline virtual std::string message(int ev) const {
    ignore<int>(ev); //suppress never used warning;
    return "Unknown error";
  }
}; // end UnknownErrorCategory

/* ErrorCode impl */
inline ErrorCode::ErrorCode() : 
    m_val(0), m_cat(&getErrorCategory<UnknownErrorCategory>()) {
} // end ErrorCode::ErrorCode()

inline ErrorCode::ErrorCode(int val, const ErrorCategory& cat) : 
    m_val(val), m_cat(&cat) {
} // end ErrorCode::ErrorCode()

template<typename Category>
inline ErrorCode ErrorCode::generate(typename Category::type e) {
  ErrorCode ec(e, getErrorCategory<Category>());
  return ec;
} // end ErrorCode::generate()

inline void ErrorCode::assign(int val, const ErrorCategory& cat) {
  m_val = val;
  m_cat = &cat;
} // end ErrorCode::assign()

template <typename Category>
inline ErrorCode& ErrorCode::operator=(typename Category::type e) {
  m_val = e;
  m_cat = &getErrorCategory<Category>();
  return *this;
} // end ErrorCode::operator=()

inline void ErrorCode::clear() {
  m_val = 0;
  m_cat = &getErrorCategory<UnknownErrorCategory>();
} // end ErrorCode::clear()

inline int ErrorCode::value() const {
  return m_val;
} // end ErrorCode::value()

inline const ErrorCategory& ErrorCode::category() const {
  return *m_cat;
} // end ErrorCode::category()

inline std::string ErrorCode::message() const {
  return category().message(value());
} // end ErrorCode::message()

inline ErrorCode::operator bool () const {
  return value() != 0;
} // end ErrorCode::operator bool()

/* ErrorCode non-member function impl */
inline bool operator<(const ErrorCode& lhs, const ErrorCode& rhs) {
  bool result = lhs.category().operator<(rhs.category());
  result = result || lhs.category().operator==(rhs.category());
  result = result || lhs.value() < rhs.value();
  return result;
} // end operator<()

template<typename charT, typename traits>
std::basic_ostream<charT, traits>& operator<< (
    std::basic_ostream<charT, traits> &os, const ErrorCode& ec) {
  return os << ec.category().name() << ':' << ec.value();
} // end operator<<()

inline bool operator==(const ErrorCode& lhs, const ErrorCode& rhs) {
  return lhs.category().operator==(rhs.category()) && lhs.value() == rhs.value();
} // end operator==()

inline bool operator!=(const ErrorCode& lhs, const ErrorCode& rhs) {
  return !(lhs == rhs);
} // end operator!=()

} /* end namespace support */

