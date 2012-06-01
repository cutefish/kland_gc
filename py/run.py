"""
run.py

Run the kernels.

"""
import logging

import configure
import common
import workdir
import pbs
import settings

logger = logging.getLogger('run')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def getCmdStr(exec_file, config_file, log_root, out_root):
    l_ret = [
        'mpirun %s %s %s %s\n' %(
            exec_file, config_file,
            log_root, out_root),
        '\n',
        'PRGM_EXIT=$?\n',
        'if [ "$PRGM_EXIT" == "0" ]; then\n',
        '\techo "success"\n',
        '\t#if work finishes successfully, delete work\n',
        '\trm %s\n' %config_file,
        'else\n',
        '\techo "failed"\n',
        'fi\n',
        'exit $PRGM_EXIT\n',
    ]
    return ''.join(s for s in l_ret)

def runWork(m_info, s_timestamp):
    work_dir = m_info['work_dir']
    script_dir = '%s/%s/scripts' %(work_dir, s_timestamp)
    config_dir = '%s/%s/partition/config' %(work_dir, s_timestamp)
    common.makeDirOrPass(script_dir)
    worklist = common.glob('%s/config[0-9]*' %config_dir)

    #create scripts
    for work in worklist:
        launch_id = common.getBasename(work).lstrip('config')
        config_file = work
        exec_file = m_info['exec']
        out_root = '%s/%s/out' %(work_dir, s_timestamp)
        log_root = '%s/%s/log/launch%s' %(work_dir, s_timestamp, launch_id)
        common.makeDirOrPass(log_root) #log_root need to be created
        pbs_setting = pbs.PBSOption()
        resources = 'nodes=%s:ppn=%s:gpus=%s,walltime=%s:00:00' %(
            settings.nodes_per_launch, settings.gpus_per_node,
            settings.gpus_per_node, settings.node_time_lim)
        d_values = {
            'out_file': '%s/terminal_out' %log_root,
            'err_file': '%s/terminal_err' %log_root,
            'res_list': resources,
        }
        pbs_setting.setOptVals(d_values)
        s_script = pbs_setting.getScriptStr(
            getCmdStr(exec_file, config_file, log_root, out_root))
        f_script = open('%s/pbslaunch%s.sh' %(script_dir, launch_id), 'w')
        f_script.write(s_script)
        f_script.close()

    #launch
    for work in worklist:
        launch_id = common.getBasename(work).rstrip('config')
        logger.info('pbs submit work: %s/pbslaunch%s.sh' %(
            script_dir, launch_id))
        pbs.submitWork('%s/pbslaunch%s.sh' %(script_dir, launch_id))

def run():
    m_info = configure.readConfig()
    work_dir = m_info['work_dir']
    exist_runs = common.glob(work_dir + '/[0-9]*_[0-9]*')
    print 'Launch: '
    print '%s: %s' %(0, 'Create new')
    print '\n'.join('%s: %s' %(i + 1, e) for i, e in enumerate(exist_runs))
    idx = int(raw_input('Select work:'))
    if idx == 0:
        s_timestamp = workdir.setup(m_info)
    else:
        s_timestamp = exist_runs[idx - 1]
        workdir.makeWorkDirs(m_info, s_timestamp)

    runWork(m_info, s_timestamp)

def main():
    run()

if __name__ == '__main__':
    main()



