#ifndef USER_CONFIG_H_
#define USER_CONFIG_H_

#include <iostream>
#include <string>
#include <vector>

class Config {
 public:
  /*** ctor/dtor ***/
  Config(std::string config_file,
         std::string log_root,
         std::string out_root);

  /*** getter ***/
  const std::vector<std::string>& temp_list() const;
  const std::vector<std::string>& cont_list() const;
  const std::vector<std::string>& channel_list() const;
  const std::string& special_channel() const;
  int temp_npts() const;
  int cont_npts() const;
  float temp_tbefore() const;
  float temp_tafter() const;
  float sample_rate() const;
  const std::string& snr_name() const;
  float snr_thr() const;
  float mad_ratio() const;
  int num_chnlThr() const;
  const std::string& log_root() const;
  const std::string& out_root() const;

  /*** repr ***/
  std::string repr();

 private:
  std::vector<std::string> m_tempList;
  std::vector<std::string> m_contList;
  std::vector<std::string> m_channelList;
  std::string m_specialChannel;
  int m_tempNpts;
  int m_contNpts;
  float m_tempTbefore;
  float m_tempTafter;
  std::string m_snrName;
  float m_snrThr;
  float m_madRatio;
  int m_numChnlThr;
  std::string m_logRoot;
  std::string m_outRoot;

  void fill(std::string key, std::string value);
};

#endif /* USER_CONFIG_H_ */
