import memory_profiler

def show_menory():
	usage=int(memory_profiler.memory_usage()[0])
	print('menory usage:',usage,'MB')
	return usage

def writelog(log,printflag = False):
    f = open('./log','a+')
    f.write(log+'\n')
    if printflag:
        print(log)