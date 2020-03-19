import os

# import memory_profiler
# def show_menory():
#     usage=int(memory_profiler.memory_usage()[0])
#     print('menory usage:',usage,'MB')
#     return usage

def writelog(log,opt,printflag = False):
    f = open(os.path.join(opt.save_dir,"log.txt"),'a+')
    f.write(log+'\n')
    if printflag:
        print(log)

def show_paramsnumber(net,opt):
    parameters = sum(param.numel() for param in net.parameters())
    parameters = round(parameters/1e6,2)
    writelog('net parameters: '+str(parameters)+'M',opt,True)

def makedirs(path):
    if os.path.isdir(path):
        print(path,'existed')
    else:
        os.makedirs(path)
        print('makedir:',path)