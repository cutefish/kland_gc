"""
pbs.py

PBS methods and script generator.

"""

import subprocess

class PBSOption:
    """PBS option. """
    _idx_string = 0
    _idx_value = 1
    def __init__(self):
        """Mirror and initialize the options."""
        self.d_options = {
            'start_time':['-a', None],
            'account':['-A', 'UT-NTNL0083'],
            'max_block_sec':['-b', None],
            'checkpoint_options':['-c', None],
            'directive_prefix':['-C', None],
            'work_dir':['-d', None],
            'root_dir':['-D', None],
            'err_file':['-e', None],
            'fault_tol':['-f', None],
            'user_hold':['-h', None],
            'interactive':['-I', None],
            'join':['-j', 'oe'],
            'keep':['-k', None],
            'res_list':['-l','nodes=10:ppn=1:gpus=3,walltime=20:00:00'],
            'mail':['-m', None],
            'mail_users':['-M', None],
            'name':['-N', None],
            'out_file':['-o', None],
            'priority':['-p', None],
            'root_repr':['-P', None],
            'queue':['-q','batch'],
            'rerunable':['-r', None],
            'shell':['-S', None],
            'array_idx':['-t', None],
            'users':['-u', None],
            'env':['-v', None],
            'export_all':['-V', None],
            'attrs':['-W', None],
            'X11':['-X', None],
            'no_id':['-z', None],
        }

    def getOptStr(self, key):
        return self.d_options[key][PBSOption._idx_string]

    def getOptVal(self, key):
        return self.d_options[key][PBSOption._idx_value]

    def setOptVal(self, key, value):
        self.d_options[key][PBSSetting.idx_value] = value

    def setOptVals(self, d_keyvalue):
        for key in d_keyvalue:
            self.setOptVal(key, d_keyvalue[key])

    def getPBSOptStr(self, key):
        return '#PBS %s %s\n' %(self.getOptStr(key),
                                self.getOptVal(key))

    def getScriptStr(self, s_cmd):
        ret = ''
        ret += '#!/bin/bash\n\n'
        for key in self.d_options:
            if (self.getOptVal(key) != None):
                ret += getPBSOptStr(key)
        ret += '### End of PBS options ###'
        ret += s_cmd
        return ret

def submitWork(script):
    cmd = 'qsub %s' %script
    subprocess.check_output(cmd.split(' '))
