
#include "support/Exception.h"
#include "user/sacio.h"
#include "user/UserErrCategory.h"

/*** SacInput ***/

/* SacInput::ctor()
 * Open file and read header
 */
SacInput::SacInput(std::string name) {
  m_ifs.open(name.c_str());
  user::throwError(m_ifs.is_open(), usererr::file_not_open, name);
  m_ifs.read((char*) &m_header, sizeof(m_header));
  user::throwError(m_ifs.good(), usererr::sac_header_error, name);
}

/* SacInput::name()
 */
std::string SacInput::name() const { return m_name; }

/* SacInput::header()
 */
SACHEAD SacInput::header() const { return m_header; }

/* SacInput::read()
 */
void SacInput::read(char* ptr, size_t pos, size_t size) {
  m_ifs.seekg(sizeof(m_header) + pos, std::ios::beg);
  m_ifs.read(ptr, size);
  user::throwError(m_ifs.good(), usererr::sac_read_error, m_name);
}

/*** SacOutput ***/

/* SacOutput::ctor()
 * Open file
 */
SacOutput::SacOutput(std::string name) {
  m_ofs.open(name.c_str());
  user::throwError(m_ofs.is_open(), usererr::file_not_open, name);
}

/* SacOutput::name()
 */
std::string SacOutput::name() const { return m_name; }

/* SacInput::header()
 */
SACHEAD SacOutput::header() const { return m_header; }


/* SacOutput::write()
 */
void SacOutput::write(char* ptr, size_t size,
                      float delta, float begin, size_t point_size) {
  //write header
  m_header = sac_null;
  m_header.npts = size / point_size;
  m_header.delta = delta;
  m_header.b = begin;
  m_header.o = 0.;
  m_header.e = begin + (m_header.npts - 1) * m_header.delta;
  m_header.iztype = IO;
  m_header.iftype = ITIME;
  m_header.leven = TRUE;
  m_ofs.write((char*)&m_header, sizeof(m_header));
  //write data
  m_ofs.write(ptr, size);
  user::throwError(m_ofs.good(), usererr::sac_write_error, m_name);
}

