//This code is adopted from thrust/system/system_error.h by Xiao Yu

#ifndef SUPPORT_ERRORCODE_H_
#define SUPPORT_ERRORCODE_H_

#include <iostream>

namespace support
{

class ErrorCode;

// Class ErrorCategory

/*! \brief 
 * 
 * The class \p ErrorCategory describes a type of error, such as OSErrCategory,
 * UserErrCategory, etc.  Users of the derived class use the get method to get
 * the only instance, otherwise the equal operator does not work as expected.
 */

/*! get method
 * \return the only instance of this category
*/
template<typename Category>
const Category& getErrorCategory();

class ErrorCategory
{
  public:

   /*! Derived Category should define a typedef a enum type
    * e.g.
    *
    * typedef EnumType errortype
    */

   /*! Destructor does nothing.
   */
   virtual ~ErrorCategory();

   virtual const char* name(void) const = 0;

   /*! \return \p ErrorCode(ev, *this).
   */
   virtual ErrorCode default_code(int ev) const;

   /*! \return A string that describes the error condition denoted by \p ev.
   */
   virtual std::string message(int ev) const = 0;

   /*! \return <tt>*this == &rhs</tt>
   */
   bool operator==(const ErrorCategory &rhs) const;

   /*! \return <tt>!(*this == rhs)</tt>
   */
   bool operator!=(const ErrorCategory &rhs) const;

   /*! \return <tt>less<const ErrorCategory*>()(this, &rhs)</tt>
    *  \note \c less provides a total ordering for pointers.
    */
   bool operator<(const ErrorCategory &rhs) const;
}; // end ErrorCategory


class UnknownErrorCategory;

// Class ErrorCode

/*! \brief 
 * 
 * The class \p ErrorCode describes an object used to hold error code values,
 * such as those originating from the operating system or other low-level
 * application program interfaces.
 */

class ErrorCode
{
  public:
    // constructors:

    /*! Effects: Constructs an object of type \p ErrorCode.
     *  \post <tt>value() == 0</tt> and 
     *  <tt>category() == * &UnknownErrorCategory::get()</tt>.
     */
    ErrorCode(void);

    /*! Effects: Constructs an object of type \p ErrorCode.
     *  \post <tt>value() == val</tt> and <tt>category() == &cat</tt>.
     */
    ErrorCode(int val, const ErrorCategory& cat);

    //named constructor:
    /*! Effects: Constructs an object of type \p ErrorCode.
     *  the value must be of corresponding enum type
     */
    template <typename Category>
    static ErrorCode generate(typename Category::type e);

    // modifiers:

    /*! \post <tt>value() == val</tt> and <tt>category() == &cat</tt>.
     */
    void assign(int val, const ErrorCategory& cat);

    /*! \post <tt>*this == make_error_code(e)</tt>.
     */
    template <typename Category>
    ErrorCode& operator=(typename Category::type e);

    /*! \post <tt>value() == 0</tt> and 
     * <tt>category() == UnknownErrorCategory::get()</tt>.
     */
    void clear(void);

    // [19.5.2.4] observers:

    /*! \return An integral value of this \p ErrorCode object.
     */
    int value(void) const;

    /*! \return An \p ErrorCategory describing the category of this \p ErrorCode object.
     */
    const ErrorCategory& category(void) const;

    /*! \return <tt>category().message(value())</tt>.
     */
    std::string message(void) const;

    /*! \return <tt>value() != 0</tt>.
     */
    operator bool (void) const;

    /*! \cond
     */
  private:
    int m_val;
    const ErrorCategory *m_cat;
    /*! \endcond
     */
}; // end ErrorCode


// [19.5.2.5] Class error_code non-member functions


/*! \return <tt>lhs.category() < rhs.category() || lhs.category() == rhs.category() && lhs.value() < rhs.value()</tt>.
 */
inline bool operator<(const ErrorCode &lhs, const ErrorCode &rhs);


/*! Effects: <tt>os << ec.category().name() << ':' << ec.value()</tt>.
 */
template <typename charT, typename traits>
  std::basic_ostream<charT,traits>&
    operator<<(std::basic_ostream<charT,traits>& os, const ErrorCode &ec);


// [19.5.4] Comparison operators


/*! \return <tt>lhs.category() == rhs.category() && lhs.value() == rhs.value()</tt>.
 */
bool operator==(const ErrorCode &lhs, const ErrorCode &rhs);

/*! \return <tt>!(lhs == rhs)</tt>
 */
inline bool operator!=(const ErrorCode &lhs, const ErrorCode &rhs);

} /* namespace support */

#include "ErrorCode.inl"

#endif /* SUPPORT_ERRORCODE_H_ */
