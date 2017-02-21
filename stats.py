import os
import time
import psutil

def memory_usage():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.get_memory_info()[0] / float(2 ** 20)
    return mem

def start_timer():
    timer = time.clock()
#     print "Start: Timer is "+str(timer)
    return timer
    
def end_timer(timer):
    "When ending a timer that was already started, returns the elapsed time in seconds"
    elapsed_time = time.clock() - timer
#     print "End: Timer is "+str(timer)
#     print "End: Elapsed "+str(elapsed_time)
    return elapsed_time
    
if __name__ == '__main__':
    print str(memory_usage())+"MB"
    timer = start_timer()
    lst = 'a'
    for i in range(1024 **2):
        lst += 'a'
    # print lst
    print str(memory_usage())+"MB"
    print str(end_timer(timer))+"s"