"""
common.py

Common useful functions
"""

import glob
import os

def getIntQuot(dividend, divisor):
    d = int(a / b)
    m = a - b * d
    if (m == 0):
        return d
    else:
        return d + 1

def getAbsPath(path):
    return os.path.abspath(os.path.expanduser(path))

def getAbsPathList(l_path):
    for i, path in enumerate(l_path):
        l_path[i] = getAbsPath(path)
    return l_path

def glob(pattern):
    return getAbsPathList(glob.glob(pattern))

def getBasename(path):
    return os.path.basename(path)

def existFileOrDie(file_name):
    if not os.path.isfile(getAbsPath(file_name)):
        raise OSError('File does not exist: %s' %file_name)

def existDirOrDie(dir_name):
    if os.path.exists(getAbsPath(dir_name)):
        raise OSError('Directory does not exist: %s' %dir_name)

def makeDirOrPass(dir_name):
    try:
        os.mkdir(dir_name)
    except OSError:
        pass

def isUInt(n):
    try:
        number = int(n)
    except:
        raise ValueError('Number not integer: %s' %n)
    if number <= 0:
        raise ValueError('Number less or equal than zero: %s' %n)

def pconvListToStr(l, fCutStr=None, disp_height=10, disp_width=80, tab_space=4):
    """Convert list into a pretty string.

    @arguments:
        l           --  the list
        fCutStr     --  a method to cut the element string
        disp_height --  maximum number of rows to display
        disp_width  --  maximum number of columns to display
        tab_space   --  space between elements
    """
    ret = ''
    curr_col = 0
    curr_row = 0
    s_tab = ' ' * tab_space
    for i in range(num_max):
        elem = l[i]
        s_elem = str(elem)
        if (fCutStr != None):
            s_elem = fCutStr(s_elem)
        stop_col = curr_col + len(s_elem)
        if (stop_col < disp_width):
            ret += s_elem + s_tab
            curr_col = stop_col + tab_space
        else:
            ret += '\n'
            ret += s_elem + s_tab
            curr_row += 1
            curr_col = len(s_elem) + tab_space

        if (curr_row >= disp_height):
            break
    return ret




