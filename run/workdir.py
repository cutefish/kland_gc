"""
workdir.py

Organizing work directories and files for launches
Work dir hierarchy:
under user work_dir
/$work_dir
    /worklist               --  work list directory
      /partition            --  file list partitions
        /temp
          /<temp_list_file>0
          ...
        /cont
          /<cont_list_file>0
            ...
      /config0              --  launch 0 config
      /config1
      ...
    /tmp                    --  temporary directory
    /log                    --  log directory
        /0
        /1
        ...
    /output                 --  output results

@method
setup()     --  set up the directory

"""

import logging
import os
import shutil
import time

from partition import getPartition

logger = logging.getLogger('workdir')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def _getTimeStr():
    time.strftime('_%y%m%d-%H:%M:%S')

def _getSubDirPath(work_dir, name, s_curr):
    return '%s/%s%s' %(work_dir, name, s_curr)

def _makeWorkListDir(m_info, s_curr):
    work_dir = m_info['work_dir']
    #worklist
    worklist_dir = _getSubDirPath(work_dir, 'worklist', s_curr)
    os.mkdir(worklist_dir)
    #partition
    partition_dir = worklist_dir + '/partition'
    os.mkdir(partition_dir)
    #partition/temp
    part_temp_dir = partition_dir + '/temp'
    os.mkdir(part_temp_dir)
    #partition/cont
    part_cont_dir = partition_dir + '/cont'
    os.mkdir(part_cont_dir)
    return worklist_dir

def _partitionList(orig_file, seg_dir, seg_size):
    #Parition \p orig_file into files of size \p seg_size
    #and put into \p seg_dir
    #Return the number of segments
    basename = os.basename(orig_file)
    f_ = open(orig_file, 'r')
    count = 0
    finished = False
    while(not finished):
        line = f_.readline()
        if line is '':
            break
        fn = '%s/%s%s' %(seg_dir, basename, count)
        f_new = open(fn, 'w')
        f_new.write(line)
        for i in range(seg_size - 1):
            line = f_.readline()
            if line is '':
                finished = True
                break
            f_new.write(line)
        f_new.close()
        count += 1
    return count

def _setupWorkList(m_info, worklist_dir, num_launch,
                   tseg_size, cseg_size):
    """Create directories, and configure files."""
    temp_list_file = m_info['temp_list_file']
    cont_list_file = m_info['cont_list_file']
    num_tSegs = _partitionList(temp_list_file,
                                worklist_dir + '/partition/temp',
                                temp_seg_size)
    _partitionList(cont_list_file,
                   worklist_dir + '/partition/cont',
                   cont_seg_size)
    f_config = open(m_info['config_file'], 'r')
    for i in range(num_launch):
        #calculate temp and cont seg index
        idx_t = i % num_tSegs
        idx_c = i / num_tSegs
        #create launch i dir
        launch_dir = '%s/%s' %(worklist_dir, i)
        os.mkdir(launch_dir)
        #create local config
        new_file = '%s/config%s' %(worklist_dir, i)
        f_new = open(new_file, 'w')
        for line in f_config:
            key, value = line.split(':')
            if key is 'temp_list_file':
                value = '%s/partition/temp/%s%s\n' %(
                    worklist_dir, temp_list_file, idx_t)
            elif key is 'cont_list_file':
                value = '%s/partition/cont/%s%s\n' %(
                    worklist_dir, cont_list_file, idx_c)
            f_new.write('%s:%s' %(key, value))

def _setupAuxDir(m_info, s_curr):
    work_dir = m_info['work_dir']
    #tmp
    tmp_dir = _getSubDirPath(work_dir, 'tmp', s_curr)
    common.makeDirOrPass(tmp_dir)
    #log
    log_dir = _getSubDirPath(work_dir, 'log', s_curr)
    common.makeDirOrPass(log_dir)
    #output
    output_dir = _getSubDirPath(work_dir, 'output', s_curr)
    common.makeDirOrPass(otuput_dir)

def _setupWorkDir(m_info, s_curr):
    work_dir = m_info['work_dir']
    try:
        os.mkdir(work_dir)
    except OSError:
        pass
    worklist_dir = _makeWorkListDir(m_info, s_curr)
    num_launch, temp_seg_size, cont_seg_size = getPartition(m_info)
    _setupWorkList(m_info, worklist_dir, num_launch,
                   temp_seg_size, cont_seg_size)
    _setupAuxDir(m_info, s_curr)
    logger.info('Work space setup done: %s' %work_dir)

def makeAuxDir(m_info, s_curr, launch_id):
    _setupAuxDir(m_info, s_curr)
    log_dir = _getSubDirPath(work_dir, 'log', s_curr)
    log_dir += str(launch_id)
    common.makeDirOrPass(log_dir)

def setup(m_info):
    """Set up work directory for launches."""
    s_curr = _getTimeStr()
    _setupWorkDir(m_info, s_curr)
    return s_curr

def getLeftWork(work_dir):
    """Return a list of non-empty worklist directories."""
    l_worklist_dirs = common.glob('%s/worklist*' %work_dir)
    l_workleft = []
    for worklist_dir in l_worklist_dirs:
        l_dirs = common.glob('%s/*' %worklist_dir)
        if (len(l_dirs) == 1):
            if (l_dirs[0] == '%s/partition' %worklist_dir):
                #this directory is empty, remove it
                print 'All work done, remove it: %s' %worklist_dir
                shutil.rmtree(worklist_dir)
            else:
                #this directory is broken, remove it
                print 'Worklist broken, missing partition info, remove it: 
                    %s' %worklist_dir
                shutil.rmtree(worklist_dir)
        else:
            l_workleft.append(worklist_dir)
    return l_workleft
