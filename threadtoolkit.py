#coding=utf-8
from multiprocessing import cpu_count
from multiprocessing import Process,Queue,Pool
from functools import partial
import traceback
import time
import os
import wml_utils as wmlu

DEFAULT_THREAD_NR=cpu_count() if cpu_count()<4 else cpu_count()-1

def fn_wraper(datas,fn,is_factory_fn=False):
    if is_factory_fn:
        fn = fn()
    res_queue = []
    print(f"Process {os.getpid()}: data nr {len(datas)}.")
    for data,i in datas:
        try:
            res = fn(data)
            res_queue.append((i,res))
        except:
            traceback.print_exc()
    print(f"Process {os.getpid()} is finished.")
    return res_queue

def par_for_each(data,fn,thread_nr=DEFAULT_THREAD_NR,is_factory_fn=False,timeout=None):
    if len(data) == 0:
        return []
    thread_nr = min(len(data),thread_nr)
    pool = Pool(thread_nr)
    data = list(zip(data,range(len(data))))
    datas = wmlu.list_to_2dlistv2(data,thread_nr)
    raw_res = list(pool.map(partial(fn_wraper,fn=fn,is_factory_fn=is_factory_fn),datas))
    pool.close()
    pool.join()

    res_data = []
    for res in raw_res:
        res_data.extend(res)
    res_data = sorted(res_data,key=lambda x:x[0])
    _,res_data = zip(*res_data)
    return res_data

def par_for_each_no_return(data,fn,thread_nr=DEFAULT_THREAD_NR):
    thread_nr = min(len(data),thread_nr)
    pool = Pool(thread_nr)
    datas = wmlu.list_to_2dlistv2(data,thread_nr)
    pool.map(fn,datas)
    pool.close()
    pool.join()
