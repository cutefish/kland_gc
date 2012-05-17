#ifndef SUPPORT_TYPE_H_
#define SUPPORT_TYPE_H_

#include <sstream>
#include <cstdlib>

namespace support {

template<typename T>
T String2Type(const std::string s) {
  std::stringstream ss(s);
  T ret;
  return (ss >> ret) ? ret : 0;
}

template<>
int String2Type<int>(const std::string s) {
  return atoi(s.c_str());
}

template<>
float String2Type<float>(std::string s) {
  return (float)atof(s.c_str());
}

template<>
double String2Type<double>(std::string s) {
  return atof(s.c_str());
}

template<typename T>
std::string Type2String(T val) {
  std::stringstream ss;
  ss << val
  return ss.str();
}

} /* namespace support */
#endif /* SUPPORT_TYPE_H_ */
