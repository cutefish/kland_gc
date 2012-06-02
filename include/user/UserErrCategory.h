
#ifndef USER_USERERRCATEGORY_H_
#define USER_USERERRCATEGORY_H_

#include <string>
#include <vector>

#include "support/Exception.h"
#include "support/ErrorCode.h"

namespace usererr {

enum UserErrEnum {
  file_not_open,
  sac_header_error,
  sac_read_error,
  sac_write_error,
  mpi_call_error,
  temp_invalid_start_time
}; // end StdErrEnum

} /* namespace errc */

inline const std::string getErrString(int ev) {
  std::vector<std::string> errs;
  errs.push_back("User: File not open. ");
  errs.push_back("User: SAC header read error. ");
  errs.push_back("User: SAC data read error. ");
  errs.push_back("User: SAC data write error. ");
  errs.push_back("User: MPI error. ");
  errs.push_back("User: Temp invalid start time. ");

  if (ev >= errs.size()) return "User: unknown_err";
  return errs[ev];
}

class UserErrCategory : public support::ErrorCategory {
 public:
  typedef usererr::UserErrEnum type;

  UserErrCategory() { }

  virtual const char* name() const {
    return "user_error";
  }

  virtual std::string message(int ev) const {
    return getErrString(ev);
  }
  
};

namespace user {

static inline void throwError(bool is_sucess, usererr::UserErrEnum error,
                              std::string message="") {
  if (is_sucess) return;
  throw support::Exception(error,
                           support::getErrorCategory<UserErrCategory>(),
                           message);
}

} /* namespace user */

#endif /* USER_USERERRCATEGORY_H_*/

