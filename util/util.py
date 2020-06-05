import os
import string
import random

def randomstr(num):
    return ''.join(random.sample(string.ascii_letters + string.digits, num))

def writelog(log,opt,printflag = False):
    f = open(os.path.join(opt.save_dir,"log.txt"),'a+')
    f.write(log+'\n')
    if printflag:
        print(log)

def makedirs(path):
    if os.path.isdir(path):
        print(path,'existed')
    else:
        os.makedirs(path)
        print('makedir:',path)

def loadtxt(path):
    f = open(path, 'r')
    txt_data = f.read()
    f.close()
    return txt_data

def savetxt(file,path):
    wf = open(path,'w')
    wf.write(file)
    wf.close()

def loadfile(path):
    rf = open(path, "rb")
    file = rf.read()
    rf.close()
    return file

def savefile(file,path):
    wf = open(path,'wb')
    wf.write(file)
    wf.close()