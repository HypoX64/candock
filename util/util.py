import os

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