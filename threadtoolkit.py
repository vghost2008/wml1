#coding=utf-8
from multiprocessing import cpu_count
from multiprocessing import Process,Queue
import traceback
import time
import os

DEFAULT_THREAD_NR=cpu_count() if cpu_count()<4 else cpu_count()-1

def fn_wraper(fn,datas,res_queue,finish_queue,is_factory_fn=False):
    if is_factory_fn:
        fn = fn()
    print(f"Process {os.getpid()}: data nr {len(datas)}.")
    for i,data in enumerate(datas):
        try:
            res = fn(data)
            res_queue.put(res)
        except:
            traceback.print_exc()
    print(f"Process {os.getpid()} is finished.")
    finish_queue.put(1)

def par_for_each(data,fn,thread_nr=DEFAULT_THREAD_NR,is_factory_fn=False,timeout=None):
    res_queue = Queue(maxsize=len(data))
    finish_queue = Queue(maxsize=thread_nr)
    thread_nr = min(len(data),thread_nr)
    block_size = len(data)//thread_nr
    last_index = 0
    process_list = []

    for i in range(thread_nr):
        try:
            if i == thread_nr-1:
                p0 = Process(target=fn_wraper, args=(fn,data[last_index:],res_queue,finish_queue,is_factory_fn))
                last_index = len(data)
            else:
                p0 = Process(target=fn_wraper, args=(fn,data[last_index:last_index+block_size],res_queue,finish_queue,is_factory_fn))
                last_index += block_size
            p0.start()
            process_list.append(p0)
        except:
            traceback.print_exc()
    for i in range(thread_nr):
        x = finish_queue.get()
        print(x)
    for p in process_list:
        p.join(timeout=timeout)

    res_data = []
    while not res_queue.empty():
        try:
            res_data.append(res_queue.get_nowait())
        except:
            traceback.print_exc()

    return res_data


