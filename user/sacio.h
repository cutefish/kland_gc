#ifndef USER_SACIO_H_
#define USER_SACIO_H_

#include <fstream>
#include <string>

#include "sac.h"

class SacInput {
 public:
  /*** ctor/dtor ***/
  SacInput(std::string name);

  /*** gettor/settor ***/
  std::string name() const;
  SACHEAD header() const;

  /*** member functions ***/
  void read(char* ptr, size_t pos, size_t size);

 private:
  string m_name;
  SACHEAD m_header;
  std::ifstream m_ifs;
};

class SacOutput {
 public:
  /*** ctor/dtor ***/
  SacOutput(std::string name);

  /*** gettor/settor ***/
  std::string name() const;
  SACHEAD header() const;

  /*** member functions ***/
  void write(char* ptr, size_t size, 
             float delta=0.01, float begin=0, size_t point_size=4);

 private:
  std::string m_name;
  SACHEAD m_header;
  std::ofstream m_ofs;
};


#endif /* USER_SACIO_H_ */
