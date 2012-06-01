"""
run.py

Run the kernels.

"""

import configure
import common
import workdir
import pbs
import settings

def getCmdStr(config_file, exec_file,
              out_root, log_root):
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
    #worklist_dir
    work_dir = m_info['work_dir']
    worklist_dir = '%s/worklist%s' %(work_dir, s_timestamp)

    #get all the work
    worklist = common.glob('%s/config[0-9]*' %worklist_dir)

    #create a temporary script dir under work_dir
    script_dir = '%s/script' %work_dir
    common.makeDirOrPass(script_dir)

    #create scripts
    for work in worklist:
        launch_id = common.getBasename(work).lstrip('config')
        config_file = work
        exec_file = m_info['exec']
        out_root = '%s/output%s' %(work_dir, s_timestamp)
        log_root = '%s/log%s/%s' %(work_dir, s_timestamp, launch_id)
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
            getCmdStr(config_file, exec_file, out_root, log_root))
        f_script = open('%s/%s' %(script_dir, work), 'w')
        f_script.write(s_script)
        f_script.close()

    #launch
    for work in worklist:
        #make sure aux dir exists
        launch_id = common.getBasename(work).lstrip('config')
        workdir.makeAuxDir(m_info, s_timestamp, launch_id)
        #launch
        pbs.submitWork('%s/%s' %(script_dir, work))

    #clean up the script dir
    os.rmdir(script_dir)

def run():
    m_info = configure.readConfig()
    work_dir = m_info['work_dir']
    l_workleft = workdir.getLeftWork(work_dir)
    worklist_idx = -1
    if (not len(l_workleft) == 0):
        print 'Work directory not empty: ' %work_dir
        print 'Left work:'
        for i, work in enumerate(l_workleft):
            print '%s: %s' %(i + 1, common.getBasename(work))
        worklist_idx = int(raw_input('Select work[0 for set up new]:')) - 1

    if (worklist_idx == -1):
        s_timestamp = workdir.setup(m_info)
    else:
        s_timestamp = common.getBasename(
            l_workleft[worklist_idx]).lstrip(worklist)

    runWork(m_info, s_timestamp)

def main():
    run()

if __name__ == '__main__':
    main()



