"""
partition.py

Calculate launche size according to configuration

@method
getPartition()
--  return number of launches, num_temp and num_cont for each launch. This
partition try to minimize the footprint for each launch.


"""

import math
import logging

import settings
import configure

secs_per_hour = 3600

logger = logging.getLogger('partition')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

#read from settings
gpus_per_node = settings.gpus_per_node
gpu_bandwidth = settings.gpu_bandwidth
sizeof_data = settings.sizeof_data
node_time_lim = settings.node_time_lim  #hours
nodes_per_launch = settings.nodes_per_launch

def _countNumLines(file_name):
    f_ = open(file_name, 'r')
    count = 0
    for line in f_:
        count += 1
    return count

def _getNumLaunch(num_temp, num_cont, num_channel, temp_npts, cont_npts):
    node_time = (num_temp * num_cont * cont_npts * temp_npts * num_channel *
                 sizeof_data)/ gpu_bandwidth / gpus_per_node #seconds
    num_node = node_time / secs_per_hour / node_time_lim
    num_launch = int(math.ceil(num_node / nodes_per_launch))
    return num_launch

#minimize the footprint of each launch
def _genWorkList(num_temp, num_cont, temp_npts, cont_npts, num_launch):
    opt_point = math.sqrt(
        (num_temp * num_cont * cont_npts) / (num_launch * temp_npts))
    tseg_size = int(opt_point)
    if (tseg_size > num_temp):
        tseg_size = num_temp
    cseg_size = num_temp * num_cont / num_launch / tseg_size;
    templist = []
    for i in range(0, num_temp, tseg_size):
        temp_start = i
        temp_end = i + tseg_size - 1
        if temp_end > num_temp:
            temp_end = num_temp - 1
        templist.append((temp_start, temp_end))
    contlist = []
    for i in range(0, num_cont, cseg_size):
        cont_start = i
        cont_end = i + cseg_size - 1
        if cont_end > num_cont:
            cont_end = num_cont - 1
        contlist.append((cont_start, cont_end))
    return templist, contlist

def genWorkList(m_info):
    temp_list_file = m_info['temp_list_file']
    num_temp = _countNumLines(temp_list_file)
    cont_list_file = m_info['cont_list_file']
    num_cont = _countNumLines(cont_list_file)
    channel_list_file = m_info['channel_list_file']
    num_channel = _countNumLines(channel_list_file)
    temp_npts = int(m_info['temp_npts'])
    cont_npts = int(m_info['cont_npts'])
    num_launch = _getNumLaunch(num_temp, num_cont,
                               num_channel, temp_npts, cont_npts)
    logger.info('approximate num_launch: %s' %num_launch)
    return _genWorkList(num_temp, num_cont, temp_npts, cont_npts, num_launch)


