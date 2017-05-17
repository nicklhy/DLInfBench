import os

def get_gpu_memory(pid=os.getpid()):
    info = os.popen('nvidia-smi').read()
    for line in info.split('\n'):
        t = line.strip().split()
        try:
            cur_pid = int(t[2])
            if cur_pid != pid:
                continue
            gpu_mem = int(t[5][:-3])
            return gpu_mem
        except:
            continue

    return 0
