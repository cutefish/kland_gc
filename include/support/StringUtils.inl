#include <iostream>
#include <sstream>

#include "StringUtils.h"

namespace support {

/* splitString() */
inline std::vector<std::string> splitString(std::string s, char delim) {
  std::stringstream ss(s);
  std::string token;
  std::vector<std::string> tokens;
  while(getline(ss, token, delim)) {
    if (token != "") tokens.push_back(token);
  }
  return tokens;
}

/* replaceString() */
inline void replaceString(std::string& original,
                          const std::string& find,
                          const std::string& replace,
                          const int& maxreplace) {
  unsigned num_replaced = 0;
  size_t pos = 0;
  while (true) {
    size_t found = original.find(find, pos);
    if (found == std::string::npos) break;
    original.replace(found, find.length(), replace);
    pos = found;
    if (maxreplace == -1) continue;
    num_replaced ++;
    if (maxreplace <= num_replaced) break;
  }
}

} /* namespace support */
