import memory_profiler

def show_menory():
	usage=int(memory_profiler.memory_usage()[0])
	print('menory usage:',usage,'MB')
	return usage