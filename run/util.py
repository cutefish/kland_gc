"""
util.py

Helper utils

@keywords
genWaveList
genChannelList

"""

import re

import common

utils = [
    'genWaveList',
    'genChannelList',
    'genConfigExample',
]

def genWaveList(root_dir, s_pattern, outfile):
    f_out = open(outfile, 'w')
    l_file = common.glob('%s/%s' %(root_dir, s_pattern))
    for f in l_file:
        f_out.write(common.getAbsPath(f))
        f_out.write('\n')
    f_out.close()

def _replaceAll(string, pattern, l_replace):
    new = string
    for r in l_replace:
        new = re.sub(pattern, r, new, 1)
    return new

def _expand(l2_option):
    l_size = []
    for l_option in l2_option:
        l_size.append(len(l_option))
    l_curr = [0] * len(l2_option)
    l_start = list(l_curr)
    ret = []
    while (True):
        entry = []
        for i, idx in enumerate(l_curr):
            entry.append(l2_option[i][idx])
        ret.append(entry)
        for i in reversed(range(len(l_curr))):
            l_curr[i] += 1
            if (l_curr[i] < l_size[i]):
                break
            l_curr[i] = 0
        if (l_start == l_curr):
            break
    return ret

def genChannelList(s_pattern, s_replace, l2_option, outfile):
    #check pattern validity
    if (len(re.findall(s_replace, s_pattern)) is not len(l2_option)):
        raise ValueError('Pattern and option does not match: %s, %s, %s'
                         %(s_pattern, s_replace, l_option))
    f_out = open(outfile, 'w')
    for option in _expand(l2_option):
        f_out.write(_replaceAll(s_pattern, s_replace, option))
        f_out.write('\n')
    f_out.close()

def genConfigExample(outfile):
    s_config = ''
    s_config += 'exec:/nics/d/home/bohong/execfile\n'
    s_config += 'work_dir:/lustre/medusa/bohong/geoproj\n'
    s_config += 'temp_list_file:/temp_list_file\n'
    s_config += 'cont_list_file:/cont_list_file\n'
    s_config += 'channel_list_file:/channel_list_file\n'
    s_config += 'special_channel:EHZ\n'
    s_config += 'temp_npts:400\n'
    s_config += 'cont_npts:8640000\n'
    s_config += 'temp_tbefore:0.5\n'
    s_config += 'temp_tafter:3.5\n'
    s_config += 'snr_thr:5\n'
    s_config += 'snr_name:wf_SNR_10_40.dat.new\n'
    s_config += 'mad_ratio:9\n'
    s_config += 'num_chnlThr:12\n'
    f_out = open(outfile, 'w')
    f_out.write(s_config)
    f_out.close()

def _chooseUtil(name):
    if name == 'genWaveList':
        root_dir = common.getAbsPath(raw_input('Root Directory:'))
        s_pattern = raw_input('Pattern:')
        outfile = common.getAbsPath(raw_input('Output File Name:'))
        genWaveList(root_dir, s_pattern, outfile)
    elif name == 'genChannelList':
        prompt = """
        Please input channel pattern.
        Example:
        /%s.%s.%s.SAC.%s//EH;ELM,ENG,LIN,HAT,RED,YOU;EN2,EH3,EHZ;bp10_40/
        /%s.%s.%s.SAC.%s/%s/EH;ELM,ENG,LIN,HAT,RED,YOU;EN2,EH3,EHZ;bp10_40/;,
        """
        print prompt
        pattern_string = raw_input('Channel Pattern:')
        splitter = pattern_string[0]
        l_pattern_str = pattern_string.lstrip(splitter).split(splitter)
        if (len(l_pattern_str) < 4):
            print 'Input format not right: %s, maybe miss last "/"' %l_pattern_str
            return
        s_pattern = l_pattern_str[0]
        s_replace = l_pattern_str[1]
        if (s_replace == ''): s_replace = '%s'
        s_options = l_pattern_str[2]
        ch_optsplit = l_pattern_str[3]
        if (ch_optsplit == ''):
            ch_optsplit0 = ';'
            ch_optsplit1 = ','
        else:
            ch_optsplit0 = l_pattern_str[3][0]
            ch_optsplit1 = l_pattern_str[3][1]
        l_option = s_options.split(ch_optsplit0)
        l2_option = []
        for option in l_option:
            l2_option.append(option.split(ch_optsplit1))
        outfile = common.getAbsPath(raw_input('Output File Name:'))
        genChannelList(s_pattern, s_replace, l2_option, outfile)
    elif name == 'genConfigExample':
        outfile = common.getAbsPath(raw_input('Output File Name:'))
        genConfigExample(outfile)
    else:
        print 'Unknown Args'

def main():
    print 'Utils:'
    for i, util in enumerate(utils):
        print '%s: %s' %(i, util)
    idx = raw_input('Select index:')
    _chooseUtil(utils[idx])

if __name__ == '__main__':
    main()
