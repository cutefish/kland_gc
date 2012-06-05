#ifndef SUPPORT_STRINGUTILS_H_
#define SUPPORT_STRINGUTILS_H_

#include <string>
#include <vector>

/*!\file StringUtils.h
 * \brief A collection of useful string util funcitons.
 */

namespace support {

/* splitString()
 * Split \p string by \p delim into vector.
 */
std::vector<std::string> splitString(std::string s, char delim);

/* replaceString()
 * Replace \p find in \p original with \p replace, the first \p maxreplace
 * occurrences are replaced.
 */
void replaceString(std::string& original,
                   const std::string& find,
                   const std::string& replace,
                   const unsigned& maxreplace=-1);

} /* namespace support */

#include "StringUtils.inl"

#endif /* SUPPORT_STRINGUTILS_H_ */
