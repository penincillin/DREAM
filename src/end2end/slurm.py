'''
This code could allocate the GPU automatically. 
Please run this code in Python 3 only!!!
''' 

import os, sys, shutil
import subprocess as sbp
from datetime import datetime

def get_all_gpu():
    all_gpus = list()
    nvidia = sbp.run(['nvidia-smi', '-L'], stdout=sbp.PIPE)
    for line in nvidia.stdout.decode('utf-8').split('\n')[:-1]:
        all_gpus.append(int(line[4]))
    return all_gpus

def get_idle_gpu(all_gpus):
    idle_gpus = list()
    for gpu_id in all_gpus:
        nvidia = sbp.run(['nvidia-smi', '-i', str(gpu_id)], stdout=sbp.PIPE)
        output = nvidia.stdout.decode('utf-8')
        if output.find('No running processes found')>=0:
            idle_gpus.append(gpu_id)
    return idle_gpus

def allocate_gpu(process_name, gpu_num):
    all_gpus = get_all_gpu()

    idle_gpus = get_idle_gpu(all_gpus)

    if gpu_num > len(idle_gpus):
        print('GPU is not enough')
    else:
        allocate_gpus = idle_gpus[:gpu_num]
        print('GPU {} is allocated'.format(','.join(map(str,allocate_gpus))))
        CUDA_command = 'CUDA_VISIBLE_DEVICES='+','.join(map(str,allocate_gpus))
        exec_command = CUDA_command + ' ' + process_name
        #print(exec_command)
        os.system(exec_command)



if __name__ == '__main__':

    if not os.path.exists('./log'):
        os.mkdir('./log/')
    log_file = './log/{}.log'.format(datetime.now().strftime('%m_%d_%H_%M'))

    process_name = 'python main.py 2>&1 | tee {}'.format(log_file)
    gpu_num = 8
    
    allocate_gpu(process_name, gpu_num)
