#coding=utf-8
import tensorflow as tf
import os
import sys
import random
import time
import wml_utils as wmlu
from multiprocessing import Process,Queue
import logging
import shutil


def get_file_name_in_ckp(name):
    names = name.split(":")
    if len(names)<2:
        return None
    name = names[-1]
    name = name.strip()
    name = name.replace("\"", "")
    return name

'''
read check point file, return the file name contented in file and the most recently file
'''
def read_check_file(filepath):
    checkpoint_files = []

    if not os.path.exists(filepath):
        return [],""

    with open(filepath, "r") as file:
        lines = file.readlines()
        recently_file = get_file_name_in_ckp(lines[0])
        for i in range(1, len(lines)):
            name = get_file_name_in_ckp(lines[i])
            if name is None:
                continue
            checkpoint_files.append(name)

    return checkpoint_files,recently_file

'''
filepath: check point file name like data.ckpt-1401.data-00000-of-00001
return the file index like 1401
'''
def file_index_of_check_file(filename):
    base_file_name = wmlu.suffix(filename)
    index = int(base_file_name.split("-")[-1])
    return index

class WEvalModel:
    '''
    tmp_dir: a dir to tmp save check point files and result
    Evaler: a evaler type, take args as initializer args if args is not none,
    evaler(ckp_file_path) have to return a value(normal a float point value) to indict which one is better and a info
    string to backup file.
    '''
    def __init__(self,Evaler,backup_dir,base_name="train_data",args=None,use_process=True,timeout=60*60):
        self.Evaler = Evaler
        self.evaler_args = args
        self.best_result = -1.
        self.best_result_time = ""
        self.best_result_t = 0.
        if not use_process:
            if self.evaler_args is not None:
                self.evaler = self.Evaler(*self.evaler_args)
            else:
                self.evaler = self.Evaler()
        else:
            self.evaler = None
        self.q = Queue()
        self.base_name = base_name
        self.backup_dir = os.path.abspath(backup_dir)
        self.history = wmlu.CycleBuffer(cap=6)
        self.timeout = timeout
        self.force_save_timeout = 60*60*2
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)

    '''
    dir_path: the check point file dir
    '''
    def __call__(self, dir_path):
        dir_path = os.path.abspath(dir_path)
        check_point_file = os.path.join(dir_path,"checkpoint")
        while True:
            check_point_files,_ = read_check_file(check_point_file)
            if self.best_result>0 and time.time()-self.best_result_t>self.force_save_timeout:
                logging.warning("Best result haven't update for more than two hours, force clean best result.")
                self.best_result = -1.

            process_nr = 0
            random.shuffle(check_point_files)

            for file in check_point_files:
                if len(wmlu.get_filenames_in_dir(dir_path=dir_path,prefix=file+".")) == 0:
                    continue
                index = file_index_of_check_file(file)
                if index in self.history:
                    continue
                logging.info("process file {}.".format(file))
                process_nr += 1
                self.history.append(index)
                filepath = os.path.join(dir_path,file)
                if self.evaler is not None:
                    result,info = self.evaler(filepath)
                else:
                    result,info = self.eval(filepath)
                if result<0.01:
                    logging.error("Unnormal result {}, ignored.".format(result))
                    continue
                if result<self.best_result:
                    logging.info("{} not the best result, best result is {}, achieved at {}, skip backup.".format(index,self.best_result, self.best_result_time))
                    continue
                print("RESULT:",self.best_result,result)
                logging.warning("New best result {}, {}.".format(file,info))
                self.best_result = result
                self.best_result_time = time.strftime("%m-%d %H:%M:%S", time.localtime())
                self.best_result_t = time.time()
                targetpath = self.backup(dir_path,file,info)
                self.save_info(dir_path,targetpath,file)

            if process_nr==0:
                logging.info("sleep for 30 seconds.")
                sys.stdout.flush()
                time.sleep(30)

    '''
    do the eval work with new process
    '''
    def eval(self,filepath):
        def do_eval(path):
            #self.q.put((10,"test"))
            #return
            try:
                if self.evaler_args is not None:
                    evaler = self.Evaler(**self.evaler_args)
                else:
                    evaler = self.Evaler()

                self.q.put(evaler(path))
            except Exception:
                self.q.put((-1.,""))

        try:
            p0 = Process(target=do_eval, args=[filepath])
            p0.start()
            p0.join(self.timeout)
            return self.q.get()
        except:
            return -1,""

    @staticmethod
    def save_info(ckp_dir,targetpath,ckp_file):
        info_file = os.path.join(ckp_dir,"best_checkpoint")
        with open(info_file,"w") as f:
            f.write(targetpath+"\n")
            f.write(ckp_file)

    @staticmethod
    def read_info(ckp_dir):
        info_file = os.path.join(ckp_dir,"best_checkpoint")

        if not os.path.exists(info_file):
            logging.error("best_checkpoint file not exists.")
            return None,None

        with open(info_file,"r") as f:
            lines = list(f.readlines())
            if len(lines)!=2:
                logging.error("error best_checkpoint file.")
                logging.info(lines)
        if len(lines)>=2:
            return lines[0].strip(),lines[1].strip()
        else:
            return None,None

    @staticmethod
    def restore_ckp(ckp_dir):
        logging.info("Try restore ckp file by evaler recoder.")
        ckp_dir = os.path.abspath(ckp_dir)
        check_point_file = os.path.join(ckp_dir,"checkpoint")

        targetpath,ckp_file = WEvalModel.read_info(ckp_dir)
        sys.stdout.flush()

        if targetpath is None or ckp_file is None:
            logging.warning("Can't restore ckp file in {}.".format(ckp_dir))
            return

        logging.info("restore file {}.".format(targetpath))
        command = "tar xvf {} -C {}".format(targetpath,ckp_dir)
        logging.info(command)
        os.system(command)
        with open(check_point_file,"w") as f:
            f.write("model_checkpoint_path: \"{}\"\n".format(ckp_file))
            f.write("all_model_checkpoint_paths: \"{}\"\n".format(ckp_file))
        sys.stdout.flush()

    '''
    backup check point file if necessary
    '''
    def backup(self,dir_path,filename,info):
        files = wmlu.get_filenames_in_dir(dir_path=dir_path,prefix=filename+".")
        index = file_index_of_check_file(filename)
        target_name = "{}_{}_{}_{}.tar.gz ".format(self.base_name,time.strftime("%y%m%d%H%M%S", time.localtime()),
                                                   index,info)
        target_path = os.path.join(self.backup_dir,target_name)
        command = "tar cvzf {} -C {} ".format(target_path,dir_path)
        for f in files:
            command += " {} ".format(f)
        logging.info("Backup check point file: {}".format(command))
        sys.stdout.flush()
        os.system(command)
        return target_path


class WEvalTarModel:
    '''
    Evaler: a evaler type, take args as initializer args if args is not none,
    evaler(ckp_file_path) have to return a value(normal a float point value) to indict which one is better and a info
    string to backup file.
    '''
    def __init__(self,Evaler,args=None,use_process=True,timeout=30*60):
        self.Evaler = Evaler
        self.evaler_args = args
        self.best_result = -1.
        self.best_result_time = ""
        self.best_result_t = 0.
        if not use_process:
            if self.evaler_args is not None:
                self.evaler = self.Evaler(*self.evaler_args)
            else:
                self.evaler = self.Evaler()
        else:
            self.evaler = None
        self.q = Queue()
        self.history = []
        self.timeout = timeout
        self.tmp_dir = "/tmp/wml_eval"
        self.best_file = None

    def extract_data(self,data_path):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir)
        os.system(f"tar xvf {data_path} -C {self.tmp_dir}")
        files = wmlu.recurse_get_filepath_in_dir(self.tmp_dir,suffix=".index")
        if len(files)==0:
            print(f"Error data {data_path}")
            return None
        file_name = wmlu.base_name(files[0])
        '''ckpt_file = os.path.join(self.tmp_dir,"checkpoint")
        with open(ckpt_file,"w") as f:
            f.write(f"model_checkpoint_path: \"{file_name}\"\n")
            f.write(f"all_model_checkpoint_paths: \"{file_name}\"\n")'''
        return os.path.join(self.tmp_dir,file_name)

    '''
    dir_path: the check point file dir
    '''
    def __call__(self, dir_path):
        dir_path = os.path.abspath(dir_path)
        while True:
            files = wmlu.recurse_get_filepath_in_dir(dir_path,suffix=".tar.gz")
            wmlu.show_list(files)
            process_nr = 0
            for file in files:
                if file in self.history:
                    continue
                ckpt_file = self.extract_data(file)
                if ckpt_file is None:
                    continue
                print("process file {}.".format(file))
                self.history.append(file)
                if self.evaler is not None:
                    result,info = self.evaler(ckpt_file)
                else:
                    result,info = self.eval(ckpt_file)
                if result<0.01:
                    print("Unnormal result {}, ignored.".format(result))
                    continue
                if result<self.best_result:
                    print("{} not the best result, best result is {}/{}, achieved at {}, skip backup.".format(file,self.best_result, self.best_file,self.best_result_time))
                    continue
                print("RESULT:",self.best_result,result)
                print("New best result {}, {}.".format(file,info))
                self.best_file = file
                self.best_result = result
                self.best_result_time = time.strftime("%m-%d %H:%M:%S", time.localtime())
                self.best_result_t = time.time()

            if process_nr==0:
                print("sleep for 30 seconds.")
                sys.stdout.flush()
                time.sleep(30)

    '''
    do the eval work with new process
    '''
    def eval(self,filepath):
        def do_eval(path):
            #self.q.put((10,"test"))
            #return
            try:
                if self.evaler_args is not None:
                    evaler = self.Evaler(**self.evaler_args)
                else:
                    evaler = self.Evaler()

                self.q.put(evaler(path))
            except Exception:
                self.q.put((-1.,""))

        try:
            p0 = Process(target=do_eval, args=[filepath])
            p0.start()
            p0.join(self.timeout)
            return self.q.get()
        except:
            return -1,""


class AutoSaveEvaler:
    def __init__(self):
        self.value = 1e5

    def __call__(self, filepath):
        self.value -= 1e-3
        return self.value,f"_1"

def auto_save(ckpt_dir,backup_dir,base_name,time_duration=2*60*60):
    eval = WEvalModel(AutoSaveEvaler,backup_dir=backup_dir,base_name=base_name,use_process=False)
    eval.force_save_timeout = time_duration
    eval(ckpt_dir)


class SearchParameters(object):
    def __init__(self,params,eval_fn,test_nr=1000,use_process=True,timeout=60*60):
        self.params = params
        self.eval_fn = eval_fn
        self.test_nr = test_nr
        self.use_process = use_process
        self.q = Queue()
        self.timeout = timeout

    def __call__(self,params=None):
        best_result = -1.0
        best_params = {}
        if params is not None:
            best_result = self.do_eval(params)
            best_params = params
        for _ in range(self.test_nr):
            params = wmlu.random_uniform_indict(self.params)
            res = self.do_eval(params)
            if res>best_result:
                best_result = res
                best_params = params
                print("New best params ",best_params," best result ",best_result)
            else:
                print("Best params ",best_params," best result ",best_result)

    def do_eval(self,params):
        if self.use_process:
            return self.eval(params)
        else:
            return self.eval_fn(params)

    '''
    do the eval work with new process
    '''
    def eval(self,params):
        def do_eval(params):
            try:
                self.q.put(self.eval_fn(params))
            except Exception:
                print("ERROR")
                self.q.put((-1.))
        try:
            p0 = Process(target=do_eval, args=[params])
            p0.start()
            p0.join(self.timeout)
            return self.q.get()
        except Exception as e:
            print("ERROR")
            return -1


