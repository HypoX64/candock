import os
import string
import random
import shutil
import json
from tensorboardX import SummaryWriter

def randomstr(num):
    return ''.join(random.sample(string.ascii_letters + string.digits, num))

def writelog(log,opt,printflag = False, tensorboard = False):
    f = open(os.path.join(opt.save_dir,"log.txt"),'a+')
    f.write(log+'\n')
    if printflag:
        print(log)
    if tensorboard:
        log = log.replace('\n', '  \n')
        opt.TBGlobalWriter.add_text('Log', log)

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

def savejson(path,data_dict):
    json_str = json.dumps(data_dict)
    f = open(path,'w+')
    f.write(json_str)
    f.close()

def copyfile(src,dst):
    try:
        if os.path.isfile(src):
            shutil.copyfile(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            print('Do not exist',src)
    except Exception as e:
        print(e)