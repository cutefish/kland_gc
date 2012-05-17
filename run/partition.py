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
    num_node = node_time / secs_per_hour / nodes_time_lim
    num_launch = num_node / nodes_per_launch
    return num_launch

#minimize the footprint of each launch
def _partition(num_temp, num_cont, temp_npts, cont_npts, num_launch):
    #this is not correct, needs to be modified
    this is not correct, needs to be modified

    opt_point = math.sqrt(
        (num_temp * num_cont * cont_npts * temp_npts) / num_launch)
    if (num_temp * temp_npts < opt_point):
        num_temp_per_launch = num_temp
    else:
        num_temp_per_launch = int(opt_point / temp_npts) + 1
    if (num_cont * cont_npts < opt_point):
        num_cont_per_launch = num_cont
    else:
        num_cont_per_launch = int(opt_point / cont_npts) + 1
    return num_temp_per_launch, num_cont_per_launch

def getPartition(m_info):
    temp_list_file = m_info['temp_list_file']
    num_temp = _countNumLines(temp_list_file)
    cont_list_file = m_info['cont_list_file']
    num_cont = _countNumLines(cont_list_file)
    channel_list_file = m_info['channel_list_file']
    num_channel = _countNumLines(channel_list_file)
    temp_npts = m_info['temp_npts']
    cont_npts = m_info['cont_npts']
    num_launch = _getNumLaunch(num_temp, num_cont,
                               num_channel, temp_npts, cont_npts)
    logger.debug('original num_launch: %s' %num_launch)
    temp_seg_size, cont_seg_size = _partition(
        num_temp, num_cont, temp_npts, cont_npts)
    num_launch = (common.getIntQuot(num_temp, temp_seg_size) *
                  common.getIntQuot(num_cont, cont_seg_size))
    logger.info('num_temp: %s, num_cont: %s, num_channel: %s, temp_npts: %s, cont_npts: %s'
                %(num_temp, num_cont, num_channel, temp_npts, cont_npts))
    logger.info('>>>num_launch: %s, temp_seg_size: %s, cont_seg_size: %s'
                %(num_launch, temp_seg_size, cont_seg_size))
    return num_launch, temp_seg_size, cont_seg_size


