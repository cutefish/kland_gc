
#ifndef USER_USERERRCATEGORY_H_
#define USER_USERERRCATEGORY_H_

#include <string>
#include <vector>

#include "support/exception/Exception.h"
#include "support/exception/ErrorCode.h"

namespace usererr {

enum UserErrEnum {
  file_not_open,
  mpi_error,
}; // end StdErrEnum

} /* namespace errc */

const std::string getErrString(int ev) {
  std::vector<std::string> errs;
  errs.push_back("User: File not open. ");
  errs.push_back("User: MPI error. ");

  if (ev >= errs.size()) return "User: unknown_err";
  return errs[ev];
}

class UserErrCategory : public support::ErrorCategory {
 public:
  typedef usererr::UserErrEnum type;

  virtual const char* name() const {
    return "user_error";
  }

  virtual std::string message(int ev) const {
    return getErrString(ev);
  }
  
};

namespace user {

static inline void throwError(bool is_sucess, usererr::UserErrEnum errno,
                              std::string message="") {
  if (is_sucess) return;
  throw support::Exception(errno,
                           support::getErrorCategory<UserErrCategory>(),
                           message);
}

} /* namespace user */

#endif /* USER_USERERRCATEGORY_H_*/

