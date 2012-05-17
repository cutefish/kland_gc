/*
 *  Copyright 2008-2011 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


/*! \file system/Exception.h
 *  \brief An exception object used to report error conditions that have an
 *         associated error code
 */

//This code is adopted from thrust/system/system_error.h
//Modified by Xiao Yu

#ifndef SUPPORT_EXCEPTION_EXCEPTION_H_
#define SUPPORT_EXCEPTION_EXCEPTION_H_

#include <stdexcept>
#include <string>

#include "support/ErrorCode.h"

namespace support
{

// Class Exception

// Class Exception overview

/*! \brief The class \p Exception describes an exception object used to report error
 *  conditions that have an associated \p ErrorCode. Such error conditions typically
 *  originate from the operating system or other low-level application program interfaces.
 *
 *  The following code listing demonstrates how to catch a \p Exception to recover
 *  from an error.
 *
 *  \code
 *
 *  #include <thrust/device_vector.h>
 *  #include <thrust/system.h>
 *  #include <thrust/sort.h>
 *
 *  void terminate_gracefully(void)
 *  {
 *    // application-specific termination code here
 *    ...
 *  }
 *
 *  int main(void)
 *  {
 *    try
 *    {
 *      thrust::device_vector<float> vec;
 *      thrust::sort(vec.begin(), vec.end());
 *    }
 *    catch(thrust::Exception e)
 *    {
 *      std::cerr << "Error inside sort: " << e.what() << std::endl;
 *      terminate_gracefully();
 *    }
 *
 *    return 0;
 *  }
 *
 *  \endcode
 *
 *  \note If an error represents an out-of-memory condition, implementations are encouraged
 *  to throw an exception object of type \p std::bad_alloc rather than \p Exception.
 */
class Exception
  : public std::runtime_error
{
  public:
    // Class Exception members
    
    /*! Constructs an object of class \p Exception.
     *  \param ec The value returned by \p code().
     *  \param what_arg A string to include in the result returned by \p what().
     *  \post <tt>code() == ec</tt>.
     *  \post <tt>std::string(what()).find(what_arg) != string::npos</tt>.
     */
    Exception(ErrorCode ec, const std::string &what_arg);

    /*! Constructs an object of class \p Exception.
     *  \param ec The value returned by \p code().
     *  \param what_arg A string to include in the result returned by \p what().
     *  \post <tt>code() == ec</tt>.
     *  \post <tt>std::string(what()).find(what_arg) != string::npos</tt>.
     */
    Exception(ErrorCode ec, const char *what_arg);

    /*! Constructs an object of class \p Exception.
     *  \param ec The value returned by \p code().
     *  \post <tt>code() == ec</tt>.
     */
    Exception(ErrorCode ec);

    /*! Constructs an object of class \p Exception.
     *  \param ev The error value used to create an \p ErrorCode.
     *  \param ecat The \p ErrorCategory used to create an \p ErrorCode.
     *  \param what_arg A string to include in the result returned by \p what().
     *  \post <tt>code() == ErrorCode(ev, ecat)</tt>.
     *  \post <tt>std::string(what()).find(what_arg) != string::npos</tt>.
     */
    Exception(int ev, const ErrorCategory &ecat, const std::string &what_arg);

    /*! Constructs an object of class \p Exception.
     *  \param ev The error value used to create an \p ErrorCode.
     *  \param ecat The \p ErrorCategory used to create an \p ErrorCode.
     *  \param what_arg A string to include in the result returned by \p what().
     *  \post <tt>code() == ErrorCode(ev, ecat)</tt>.
     *  \post <tt>std::string(what()).find(what_arg) != string::npos</tt>.
     */
    Exception(int ev, const ErrorCategory &ecat, const char *what_arg);

    /*! Constructs an object of class \p Exception.
     *  \param ev The error value used to create an \p ErrorCode.
     *  \param ecat The \p ErrorCategory used to create an \p ErrorCode.
     *  \post <tt>code() == ErrorCode(ev, ecat)</tt>.
     */
    Exception(int ev, const ErrorCategory &ecat);

    /*! Destructor does not throw.
     */
    virtual ~Exception(void) throw () {};
    
    /*! Returns an object encoding the error.
     *  \return <tt>ec</tt> or <tt>ErrorCode(ev, ecat)</tt>, from the
     *          constructor, as appropriate.
     */
    const ErrorCode& code(void) const throw();

    /*! Returns a human-readable string indicating the nature of the error.
     *  \return a string incorporating <tt>code().message()</tt> and the
     *          arguments supplied in the constructor.
     */
    const char *what(void) const throw();

    /*! \cond
     */
  private:
    ErrorCode          m_error_code;

    /*! \endcond
     */
}; // end Exception

} /* namespace support */

#include "../lib/support/Exception.inl"

#endif /* SUPPORT_EXCEPTION_EXCEPTION_H_ */
