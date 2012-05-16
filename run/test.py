import os

root_dir = '/tmp'
temp_dir = root_dir + '/Template'
cont_dir = root_dir + '/Continuous'
num_temp = 200
num_cont = 8
num_ch = 18


def makeDirOrPass(dir_name):
    try:
        os.mkdir(dir_name)
    except OSError:
        pass

def main():
    makeDirOrPass(root_dir)
    makdDirOrPass(temp_dir)
    makdDirOrPass(cont_dir)

    for i in range(num_temp):
        d = '%s/%s' %(temp_dir,i)
        makeDirOrPass(d)
        for j in range(num_ch):
            f = '%s/ch%s'(d, j)
            fh = open(f, 'w')
            fh.close()

    for i in range(num_cont):
        d = '%s/%s' %(cont_dir,i)
        makeDirOrPass(d)
        for j in range(num_ch):
            f = '%s/ch%s'(d, j)
            fh = open(f, 'w')
            fh.close()

if __name__ == '__main__':
    main()
