"""
configure.py

Get useful information from config file
Information includes
Working directory
The list file of template waveform directories;
The list file of continuous waveform directories;
The list file of channels;
Number of points for each template file;
Number of points for each continuous file;

@method
readConfigure() -- read configuration from a file or input

"""
import common

l_info = [
    'exec',
    'config_file',
    'work_dir',
    'temp_list_file',
    'cont_list_file',
    'channel_list_file',
    'special_channel',
    'temp_npts',
    'cont_npts',
    'temp_tbefore',
    'temp_tafter',
    'snr_thr',
    'mad_ratio',
]

def _getConfigStr(m_info):
    ret = ''
    for k, v in m_info:
        ret += '%s:%s\n' %(k,v)
    return ret

def _readConfigFile(config_file):
    m_info = {}
    f_config = open(config_file, 'r')
    for line in f_config:
        s_line = line.strip()
        (key, value) = s_line.split(':')
        if key in l_info:
            m_info[key] = value
    print '\nConfigurations:\n'
    print _getConfigStr(m_info)
    return m_info

def _checkConfig(m_info):
    common.existFileOrDie(m_info['temp_list_file'])
    common.existFileOrDie(m_info['cont_list_file'])
    common.existFileOrDie(m_info['channel_list_file'])
    common.isUInt(m_info['temp_npts'])
    common.isUInt(m_info['cont_npts'])


def readConfig():
    config_file = common.getAbsPath(raw_input('Configure file:'))
    m_info = _readConfigFile(config_file)
    m_info['config_file'] = config_file
    return m_info

def main():
    readConfig()

if __name__ == '__main__':
    main()

