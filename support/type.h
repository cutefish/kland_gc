
template<typename T>
T String2Type(std::string s) {
  T ret;
  return ret;
}

template<>
int String2Type<int>(std::string s) {
}

template<>
float String2Type<float>(std::string s) {
}

template<>
double String2Type<double>(std::string s) {
}
