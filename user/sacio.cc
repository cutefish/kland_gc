#include "sacio.h"

#include "support/Exception.h"
#include "UserErrCategory.h"

/*** SacInput ***/

/* SacInput::ctor()
 * Open file and read header
 */
SacInput::SacInput(std::string name) {
  m_ifs.open(name.c_str());
  user::throwError(m_ifs.is_open(), usererr::file_not_open, name);
  m_ifs.read((char*) &m_header, sizeof(m_header));
  user::throwError(m_ifs.is_good(), usererr::sac_header_error, name);
}

/* SacInput::name()
 */
inline std::string SacInput::name() const { return m_name; }

/* SacInput::header()
 */
inline SACHEAD SacInput::header() const { return m_header; }

/* SacInput::read()
 */
void SacInput::read(char* ptr, size_t pos, size_t size) {
  m_ifs.seekg(sizeof(m_header) + pos, std::ios::beg);
  m_ifs.read(ptr, size);
  user::throwError(m_ifs.is_good(), usererr::sac_read_error, name);
}

/*** SacOutput ***/

/* SacOutput::ctor()
 * Open file
 */
SacOutput::SacOutput() {
  m_ofs.open(name_.c_str());
  user::throwError(m_ofs.is_open(), usererr::file_not_open, name);
}

/* SacOutput::name()
 */
inline std::string SacOutput::name() const { return m_name; }

/* SacInput::header()
 */
inline SACHEAD SacOutput::header() const { return m_header; }


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
  m_header.e = b0 + (npts - 1) * header.delta;
  m_header.iztype = IO;
  m_header.iftype = ITIME;
  m_header.leven = TRUE;
  m_ofs.write((char*)&m_header, sizeof(m_header));
  //write data
  m_ofs.write(ptr, size);
  user::throwError(m_ofs.is_good(), usererr::sac_write_error, name);
}

