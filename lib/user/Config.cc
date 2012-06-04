#include <fstream>
#include <sstream>

#include "user/Config.h"
#include "user/UserErrCategory.h"
#include "support/Exception.h"
#include "support/Type.h"
#include "support/StringUtils.h"

/* readList() */
void readList(std::string list_file,
              std::vector<std::string>& list) {
  std::ifstream if_list;
  std::string line;
  if_list.open(list_file.c_str());
  user::throwError(if_list.is_open(), usererr::file_not_open, list_file);
  while(if_list.good()) {
    getline(if_list, line);
    if (line != "") list.push_back(line);
  }
}

/* fill() */
void Config::fill(std::string key, std::string value) {
  if (key == "temp_list_file")
    readList(value, m_tempList);
  else if (key == "cont_list_file")
    readList(value, m_contList);
  else if (key == "channel_list_file")
    readList(value, m_channelList);
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
  else if (key == "snr_name")
    m_snrName = value;
  else if (key == "snr_thr")
    m_snrThr = support::String2Type<float>(value);
  else if (key == "mad_ratio")
    m_madRatio = support::String2Type<float>(value);
  else if (key == "num_chnlThr")
    m_numChnlThr = support::String2Type<int>(value);
}

/* Config() */
Config::Config(std::string config_file,
               std::string log_root,
               std::string out_root) {
  std::ifstream if_config;
  std::string line;
  if_config.open(config_file.c_str());
  user::throwError(if_config.is_open(), 
                   usererr::file_not_open, config_file);

  while(if_config.good()) {
    getline(if_config, line);
    if (line == "") continue;
    std::vector<std::string> tokens = support::splitString(line, ':');
    fill(tokens[0], tokens[1]);
  }

  m_logRoot = log_root;
  m_outRoot = out_root;
}

/* repr() */
std::string Config::repr() {
  std::stringstream ss;
  ss << "Config: " << '\n';
  ss << "Temp list size: " << m_tempList.size() << '\n';
  ss << "Cont list size: " << m_contList.size() << '\n';
  ss << "Channel list size: " << m_channelList.size() << '\n';
  ss << "Special channel: " << m_specialChannel << '\n';
  ss << "temp_npts: " << m_tempNpts << '\n';
  ss << "cont_npts: " << m_contNpts << '\n';
  ss << "temp_tbefore: " << m_tempTbefore << '\n';
  ss << "temp_tafter: " << m_tempTafter << '\n';
  ss << "snr_name: " << m_snrName << '\n';
  ss << "snr_thr: " << m_snrThr << '\n';
  ss << "mad_ratio: " << m_madRatio << '\n';
  ss << "num_chanlThr: " << m_madRatio << '\n';
  ss << "log_root: " << m_logRoot << '\n';
  ss << "out_root: " << m_outRoot << '\n';
  ss << '\n';
  return ss.str();
}

/* getter */
const std::vector<std::string>& Config::temp_list() const { 
  return  m_tempList; 
}
const std::vector<std::string>& Config::cont_list() const { 
  return m_contList; 
}
const std::vector<std::string>& Config::channel_list() const { 
  return m_channelList; 
}
const std::string& Config::special_channel() const { 
  return m_specialChannel; 
}
int Config::temp_npts() const { return m_tempNpts; }
int Config::cont_npts() const { return m_contNpts; }
float Config::temp_tbefore() const { return m_tempTbefore; }
float Config::temp_tafter() const { return m_tempTafter; }
float Config::sample_rate() const { 
  return m_tempNpts / (m_tempTbefore + m_tempTafter); 
}
const std::string& Config::snr_name() const { return m_snrName; }
float Config::snr_thr() const { return m_snrThr; }
float Config::mad_ratio() const { return m_madRatio; }
int Config::num_chnlThr() const { return m_numChnlThr; }
const std::string& Config::log_root() const { return m_logRoot; }
const std::string& Config::out_root() const { return m_outRoot; }

