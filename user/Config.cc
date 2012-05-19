#include <fstream>
#include <sstream>

#include "Config.h"
#include "Category.h"
#include "support/Exception.h"
#include "support/Type.h"

/* splitString() */
inline void splitString(std::string s, std::vector<std::string>& tokens,
                 char delim) {
  stringstream ss(s);
  string token;
  while(getline(ss, token, delim)) {
    if (token != "") tokens.push_back(token);
  }
}

/* readList() */
inline void readList(std::string list_file,
              std::vector<std::string>& list) {
  std::ifstream if_list;
  std::string line;
  if_list.open(list_file);
  user::throwError(if_list.is_open(), usererr::file_not_open, list_file);
  while(if_list.good()) {
    getline(if_list, line);
    if (line != "") list.push_back(line);
  }
}

/* fill() */
inline void Config::fill(std::string key, std::string value) {
  if (key == "temp_list_file")
    readList(key, m_tempList);
  else if (key == "cont_list_file")
    readList(key, m_contList);
  else if (key == "channel_list_file")
    readList(key, m_channelList);
  else if (key == "special_channel")
    m_specialChannel = value;
  else if (key == "temp_npts")
    m_tempNpts = support::String2Type<int>(value);
  else if (key == "cont_npts")
    m_contNpts = support::String2Type<int>(value);
  else if (key == "temp_tbefore")
    m_tempTbefore = support::String2Type<float>(value);
  else if (key == "temp_tafter")
    m_tempTafter = support::String2Type<float>(value);
  else if (key == "snr_thr")
    m_snrThr = support::String2Type<float>(value);
  else if (key == "mad_ratio")
    m_madRatio = support::String2Type<float>(value);
}

/* Config() */
inline Config::Config(std::string config_file,
               std::string log_root,
               std::string out_root) {
  std::ifstream if_config;
  std::string line;
  if_config.open(config_file);
  user::throwError(if_list.is_open(), usererr::file_not_open, list_file);

  while(if_config.good()) {
    getline(if_config, line);
    std::vector<std::string> tokens;
    if (line == "") continue;
    splitString(line, tokens, ':');
    fill(tokens[0], tokens[1])
  }

  m_logRoot = log_root;
  m_outRoot = out_root;
}

/* print() */
inline void Config::print(ostream& out) {
  out << "Temp list size: " << m_tempList.size() << '\n';
  out << "Cont list size: " << m_contList.size() << '\n';
  out << "Channel list size: " << m_channelList.size() << '\n';
  out << "Special channel: " << m_specialChannel << '\n';
  out << "temp_npts: " << m_tempNpts << '\n';
  out << "cont_npts: " << m_contNpts << '\n';
  out << "temp_tbefore: " << m_tempTbefore << '\n';
  out << "temp_tafter: " << m_tempTafter << '\n';
  out << "snr_thr: " << m_snrThr << '\n';
  out << "mad_ratio: " << m_madRatio << '\n';
  out << "log_root: " << m_logRoot << '\n';
  out << "out_root: " << m_outRoot << '\n';
}
