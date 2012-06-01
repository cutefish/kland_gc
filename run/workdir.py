"""
workdir.py

Organizing work directories and files for launches
Work dir hierarchy:
under user defined <work_dir>
/<work_dir>
--/<timestamp>               --  work list directory
----/partition            --  file list partitions
------/temp
--------/temp0
        ...
------/cont
--------/cont0
        ...
------/config
--------/config0              --  launch 0 config
--------/config1
        ...
----/log                    --  log directory
------/launch0
--------/<userlog>0
--------/<userlog>1
        ...
----/out                    --  output results
------/<template0>
------/<template1>
      ...

@method
setup()     --  set up the directory

"""

import logging
import os
import shutil
import time

from worklist import genWorkList

logger = logging.getLogger('workdir')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def _getTimeStr():
    time.strftime('%y%m%d_%H%M%S')

def makeWorkDirs(m_info, s_curr):
    work_dir = m_info['work_dir']
    common.makeDirOrPass('%s/%s/partition/temp' %(work_dir, s_curr))
    common.makeDirOrPass('%s/%s/partition/cont' %(work_dir, s_curr))
    common.makeDirOrPass('%s/%s/partition/config' %(work_dir, s_curr))
    common.makeDirOrPass('%s/%s/log' %(work_dir, s_curr))
    common.makeDirOrPass('%s/%s/out' %(work_dir, s_curr))

def _getLinesFromFile(filename):
    fh = open(filename, 'r')
    ret = []
    while(True):
        line = fh.readline()
        if line == '':
            break
        ret.append(line)
    return ret

def _setupWorks(m_info, s_curr):
    partition_dir = '%s/%s/partition' %(
        m_info['work_dir'], s_curr)
    temp_dir = partition_dir + '/temp'
    cont_dir = partition_dir + '/cont'
    config_dir = partition_dir + '/config'
    #read list
    temp_list_file = m_info['temp_list_file']
    templist = _getLinesFromFile(temp_list_file)
    cont_list_file = m_info['temp_list_file']
    contlist = _getLinesFromFile(cont_list_file)
    #partition works
    temp_works, cont_works = genWorkList(m_info)
    # temp
    for i, work in enumerate(temp_works):
        start = work[0]
        end = work[1]
        fh = open(temp_dir + 'temp' + str(i), 'w')
        for j in range(start, end + 1):
            fh.write(templist[i])
        fh.close()
    # cont
    for i, work in enumerate(cont_works):
        start = work[0]
        end = work[1]
        fh = open(cont_dir + 'cont' + str(i), 'w')
        for j in range(start, end + 1):
            fh.write(contlist[i])
        fh.close()
    # config
    configs = _getLinesFromFile(m_info['config_file'])
    for i, twork in enumerate(temp_works):
        for j, cwork in enumerate(cont_works):
            idx = i * len(cont_works) + j
            fh = open(config_dir + 'config' + str(idx), 'w')
            for l in configs:
                key, value = l.strip().split(':')
                if key == 'temp_list_file':
                    value = '%s/temp%s' %(temp_dir, i)
                elif key == 'cont_list_file':
                    value = '%s/cont%s' %(cont_dir, j)
                fh.write('%s:%s\n' %(key, value))

def setup(m_info):
    """Set up work directory for launches."""
    s_curr = _getTimeStr()
    makeWorkDirs(m_info, s_curr)
    _setupWorks(m_info, s_curr)
    return s_curr
