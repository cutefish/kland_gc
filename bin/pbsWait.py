import time
import subprocess

def waitAndCheck(cmd, wait_time):
    while(True):
        time.sleep(wait_time);
        subprocess.call(cmd.split(' '))

def main():
    cmd = 'qstat -u bohong'
    waitAndCheck(cmd, 3)

if __name__ == '__main__':
    main()
