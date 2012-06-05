#ifndef SUPPORT_STDERRCATEGORY_H_
#define SUPPORT_STDERRCATEGORY_H_

#include <cstring>

#include "std_errno.h"
#include "ErrorCode.h"

namespace support {

namespace errc {

enum StdErrEnum {
  address_family_not_supported       = errc::eafnosupport,
  address_in_use                     = errc::eaddrinuse,
  address_not_available              = errc::eaddrnotavail,
  already_connected                  = errc::eisconn,
  argument_list_too_long             = errc::e2big,
  argument_out_of_domain             = errc::edom,
  bad_address                        = errc::efault,
  bad_file_descriptor                = errc::ebadf,
  bad_message                        = errc::ebadmsg,
  broken_pipe                        = errc::epipe,
  connection_aborted                 = errc::econnaborted,
  connection_already_in_progress     = errc::ealready,
  connection_refused                 = errc::econnrefused,
  connection_reset                   = errc::econnreset,
  cross_device_link                  = errc::exdev,
  destination_address_required       = errc::edestaddrreq,
  device_or_resource_busy            = errc::ebusy,
  directory_not_empty                = errc::enotempty,
  executable_format_error            = errc::enoexec,
  file_exists                        = errc::eexist,
  file_too_large                     = errc::efbig,
  filename_too_long                  = errc::enametoolong,
  function_not_supported             = errc::enosys,
  host_unreachable                   = errc::ehostunreach,
  identifier_removed                 = errc::eidrm,
  illegal_byte_sequence              = errc::eilseq,
  inappropriate_io_control_operation = errc::enotty,
  interrupted                        = errc::eintr,
  invalid_argument                   = errc::einval,
  invalid_seek                       = errc::espipe,
  io_error                           = errc::eio,
  is_a_directory                     = errc::eisdir,
  message_size                       = errc::emsgsize,
  network_down                       = errc::enetdown,
  network_reset                      = errc::enetreset,
  network_unreachable                = errc::enetunreach,
  no_buffer_space                    = errc::enobufs,
  no_child_process                   = errc::echild,
  no_link                            = errc::enolink,
  no_lock_available                  = errc::enolck,
  no_message_available               = errc::enodata,
  no_message                         = errc::enomsg,
  no_protocol_option                 = errc::enoprotoopt,
  no_space_on_device                 = errc::enospc,
  no_stream_resources                = errc::enosr,
  no_such_device_or_address          = errc::enxio,
  no_such_device                     = errc::enodev,
  no_such_file_or_directory          = errc::enoent,
  no_such_process                    = errc::esrch,
  not_a_directory                    = errc::enotdir,
  not_a_socket                       = errc::enotsock,
  not_a_stream                       = errc::enostr,
  not_connected                      = errc::enotconn,
  not_enough_memory                  = errc::enomem,
  not_supported                      = errc::enotsup,
  operation_canceled                 = errc::ecanceled,
  operation_in_progress              = errc::einprogress,
  operation_not_permitted            = errc::eperm,
  operation_not_supported            = errc::eopnotsupp,
  operation_would_block              = errc::ewouldblock,
  owner_dead                         = errc::eownerdead,
  permission_denied                  = errc::eacces,
  protocol_error                     = errc::eproto,
  protocol_not_supported             = errc::eprotonosupport,
  read_only_file_system              = errc::erofs,
  resource_deadlock_would_occur      = errc::edeadlk,
  resource_unavailable_try_again     = errc::eagain,
  result_out_of_range                = errc::erange,
  state_not_recoverable              = errc::enotrecoverable,
  stream_timeout                     = errc::etime,
  text_file_busy                     = errc::etxtbsy,
  timed_out                          = errc::etimedout,
  too_many_files_open_in_system      = errc::enfile,
  too_many_files_open                = errc::emfile,
  too_many_links                     = errc::emlink,
  too_many_symbolic_link_levels      = errc::eloop,
  value_too_large                    = errc::eoverflow,
  wrong_protocol_type                = errc::eprototype,
}; // end StdErrEnum

} /* namespace errc */

class StdErrCategory : public ErrorCategory {
 public:
  typedef errc::StdErrEnum type;

  virtual const char* name() const {
    return "std_error";
  }

  virtual std::string message(int ev) const {
    static const std::string unknown_err("Unknown standard error");
//    char buf[256];
//    //using strerror_r for thread-safe, not provided on windows
//    return (strerror_r(ev, buf, 256) == 0)? 
//        std::string(buf) : unknown_err;
    const char* c_str = std::strerror(1);
    return c_str ? std::string(c_str) : unknown_err;
  }
  
};

} /* namespace support */

#endif /* SUPPORT_STDERRCATEGORY_H_ */

