# -*- coding: utf-8 -*-
import tensorflow as tf
import logging
import wml_utils as wmlu
import numpy as np
from object_detection2.modeling.meta_arch.build import build_model
from object_detection2.data.dataloader import DataLoader
import time
import weakref
import wnn
import sys
import os
import wml_tfutils as wmlt
import wsummary
import object_detection2.metrics.toolkit as mt
from object_detection2.standard_names import *
from tensorflow.python import debug as tfdbg

FLAGS = tf.app.flags.FLAGS
CHECK_POINT_FILE_NAME = "data.ckpt"

class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:

    .. code-block:: python

        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        hook.after_train()

    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer when the hook is
            registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self,cfg):
        self._hooks = []
        self.cfg = cfg
        self.full_trace = False

    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int=0, max_iter: int=sys.maxsize):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        try:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
        except Exception:
            logger.exception("Exception during training:")
        finally:
            self.after_train()

    def test(self, start_iter: int=0, max_iter: int=sys.maxsize):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        evaler = mt.COCOEvaluation(num_classes=self.cfg.DATASETS.NUM_CLASSES,
                                   mask_on=self.cfg.MODEL.MASK_ON)

        try:
            self.before_train()
            while True:
                self.before_step()
                results = self.run_eval_step()
                self.model.doeval(evaler,results)
                if self.step %self.show_eval_results_step == 0:
                    evaler.show()
                self.after_step()
        except Exception:
            logger.exception("Exception during training:")
        finally:
            self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, cfg,model, data,gpus=None,inference=False,debug_tf=False):
        """
        Args:
            model: a objectdetection Module. Takes a data from data_loader and returns a
                dict of losses.
            data: when inference is False, data is a tf.Dataset.Iterator Contains data to be used to call model.
                when inference is True, data is a tf.Tensor
            gpus:list(int), when train on multi gpus, it's the gpus index to be used, or Nonr for train on single gpu
            inference: whether in inference mode
            debug_tf: whether debug tensorflow
        """
        super().__init__(cfg=cfg)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
        datefmt = "%H:%M:%S")

        self.model = model
        self.data = data
        self.loss_dict = {}
        self.sess = None
        self.global_step = tf.train.get_or_create_global_step()
        self.log_step = cfg.GLOBAL.LOG_STEP if model.is_training else cfg.GLOBAL.LOG_STEP//2
        self.save_step = cfg.GLOBAL.SAVE_STEP
        self.show_eval_results_step = 100
        self.step = 1
        self.total_loss = None
        self.variables_to_train = None
        self.summary = None
        self.summary_writer = None
        self.saver = None
        self.res_data = None
        self.ckpt_dir = os.path.join(self.cfg.ckpt_dir,self.cfg.GLOBAL.PROJ_NAME)
        self.gpus = gpus
        self.debug_tf = debug_tf #example filter: r -f has_inf_or_nan
        if inference:
            assert not model.is_training,"Error training statu"
        if model.is_training:
            self.log_dir = os.path.join(self.cfg.log_dir,self.cfg.GLOBAL.PROJ_NAME+"_log")
            self.name = f"{self.cfg.GLOBAL.PROJ_NAME} trainer"
        else:
            self.log_dir = os.path.join(self.cfg.log_dir,self.cfg.GLOBAL.PROJ_NAME+"_eval_log")
            self.name = f"{self.cfg.GLOBAL.PROJ_NAME} evaler"
        self.input_data = None
        print("Log dir",self.log_dir)
        print("ckpt dir",self.ckpt_dir)
        try:
            if tf.gfile.Exists(self.log_dir):
                tf.gfile.DeleteRecursively(self.log_dir)
        except:
            pass
        if inference:
            self.build_inference_net()
        elif not self.model.is_training or self.gpus is None:
            self.build_net()
        else:
            self.build_net_run_on_multi_gpus()

        self.res_data_for_eval = self.res_data
        self.res_data_for_eval.update(self.input_data)


    def __del__(self):
        if self.sess is not None:
            if self.saver is not None:
                self.saver.save(self.sess, os.path.join(self.ckpt_dir, CHECK_POINT_FILE_NAME), global_step=self.step)
            self.sess.close()
        if self.summary_writer is not None:
            self.summary_writer.close()

    def build_inference_net(self):
        if not os.path.exists(self.log_dir):
            wmlu.create_empty_dir(self.log_dir)
        if not os.path.exists(self.ckpt_dir):
            wmlu.create_empty_dir(self.ckpt_dir)
        '''
        When inference, self.data is just a tensor
        '''
        data = {IMAGE:self.data}
        DataLoader.detection_image_summary(data,name="data_source")
        self.input_data = data
        '''if self.cfg.GLOBAL.DEBUG:
            data[IMAGE] = tf.Print(data[IMAGE],[tf.shape(data[IMAGE]),data[ORG_HEIGHT],data[ORG_WIDTH],data[HEIGHT],data[WIDTH]],summarize=100,
                                   name="XXXXX")'''
        self.res_data,loss_dict = self.model.forward(data)
        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.allow_growth = True
        self.top_variable_name_scope = "Model"

        print("Update ops.")
        wmlu.show_list([x.name for x in tf.get_collection(tf.GraphKeys.UPDATE_OPS)])

    def build_net(self):
        if not os.path.exists(self.log_dir):
            wmlu.create_empty_dir(self.log_dir)
        if not os.path.exists(self.ckpt_dir):
            wmlu.create_empty_dir(self.ckpt_dir)
        with tf.device(":/cpu:0"):
            data = self.data.get_next()
        DataLoader.detection_image_summary(data,name="data_source")
        self.input_data = data
        '''if self.cfg.GLOBAL.DEBUG:
            data[IMAGE] = tf.Print(data[IMAGE],[tf.shape(data[IMAGE]),data[ORG_HEIGHT],data[ORG_WIDTH],data[HEIGHT],data[WIDTH]],summarize=100,
                                   name="XXXXX")'''
        self.res_data,loss_dict = self.model.forward(data)
        if self.model.is_training:
            for k,v in loss_dict.items():
                tf.summary.scalar(f"loss/{k}",v)
                v = tf.cond(tf.logical_or(tf.is_nan(v),tf.is_inf(v)),lambda : tf.zeros_like(v),lambda:v)
                tf.losses.add_loss(v)

        self.loss_dict = loss_dict
        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.top_variable_name_scope = "Model"

        if self.model.is_training:
            steps = self.cfg.SOLVER.STEPS
            print("Train steps:",steps)
            lr = wnn.build_learning_rate(self.cfg.SOLVER.BASE_LR,global_step=self.global_step,
                                     lr_decay_type="piecewise",steps=steps,decay_factor=0.1,
                                     warmup_steps=self.cfg.SOLVER.WARMUP_ITERS)
            tf.summary.scalar("lr", lr)
            opt = wnn.str2optimizer("Momentum", lr,momentum=0.9)
            self.max_train_step = steps[-1]
            self.train_op,self.total_loss,self.variables_to_train = wnn.nget_train_op(self.global_step,optimizer=opt,
                                                                                      clip_norm=self.cfg.SOLVER.CLIP_NORM)
            print("variables to train:")
            wmlu.show_list(self.variables_to_train)
            for v in self.variables_to_train:
                wsummary.histogram_or_scalar(v,v.name[:-2])

            self.saver = tf.train.Saver()
            tf.summary.scalar("total_loss",self.total_loss)

        self.summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("Update ops.")
        wmlu.show_list([x.name for x in tf.get_collection(tf.GraphKeys.UPDATE_OPS)])

    def build_net_run_on_multi_gpus(self):
        if not os.path.exists(self.log_dir):
            wmlu.create_empty_dir(self.log_dir)
        if not os.path.exists(self.ckpt_dir):
            wmlu.create_empty_dir(self.ckpt_dir)
        '''if self.cfg.GLOBAL.DEBUG:
            data[IMAGE] = tf.Print(data[IMAGE],[tf.shape(data[IMAGE]),data[ORG_HEIGHT],data[ORG_WIDTH],data[HEIGHT],data[WIDTH]],summarize=100,
                                   name="XXXXX")'''
        all_loss_dict = {}
        steps = self.cfg.SOLVER.STEPS
        print("Train steps:",steps)
        lr = wnn.build_learning_rate(self.cfg.SOLVER.BASE_LR,global_step=self.global_step,
                                     lr_decay_type="piecewise",steps=steps,decay_factor=0.1,
                                     warmup_steps=self.cfg.SOLVER.WARMUP_ITERS)
        tf.summary.scalar("lr",lr)
        self.max_train_step = steps[-1]
        opt = wnn.str2optimizer("Momentum", lr,momentum=0.9)
        tower_grads = []
        for i in range(len(self.gpus)):
            scope = tf.get_variable_scope()
            if i>0:
                #scope._reuse = tf.AUTO_REUSE
                scope.reuse_variables()
            with tf.device(f"/gpu:{i}"):
                with tf.device(":/cpu:0"):
                    data = self.data.get_next()
                    DataLoader.detection_image_summary(data,name=f"data_source{i}")
                self.input_data = data
                with tf.name_scope(f"GPU{self.gpus[i]}"):
                    self.res_data,loss_dict = self.model.forward(data)
                loss_values = []
                for k,v in loss_dict.items():
                    all_loss_dict[k+f"_stage{i}"] = v
                    tf.summary.scalar(f"loss/{k}",v)
                    v = tf.cond(tf.logical_or(tf.is_nan(v), tf.is_inf(v)), lambda: tf.zeros_like(v), lambda: v)
                    loss_values.append(v)

                scope._reuse = tf.AUTO_REUSE
                grads,total_loss,variables_to_train = wnn.nget_train_opv3(optimizer=opt,loss=loss_values)
                tower_grads.append(grads)
        avg_grads = wnn.average_grads(tower_grads,clip_norm=self.cfg.SOLVER.CLIP_NORM)
        opt0 = wnn.apply_gradientsv3(avg_grads, self.global_step, opt)
        opt1 = wnn.get_batch_norm_ops()
        self.train_op = tf.group(opt0, opt1)

        self.total_loss,self.variables_to_train = total_loss,variables_to_train

        self.loss_dict = all_loss_dict

        config = tf.ConfigProto(allow_soft_placement=True)
        #config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        if self.debug_tf:
            self.sess = tfdbg.LocalCLIDebugWrapperSession(self.sess)

        print("variables to train:")
        wmlu.show_list(self.variables_to_train)
        for v in self.variables_to_train:
            wsummary.histogram_or_scalar(v,v.name[:-2])

        self.saver = tf.train.Saver()
        tf.summary.scalar("total_loss",self.total_loss)

        self.summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("Update ops.")
        wmlu.show_list([x.name for x in tf.get_collection(tf.GraphKeys.UPDATE_OPS)])

    def resume_or_load(self,ckpt_path=None,sess=None,**kwargs):
        if ckpt_path is None:
            ckpt_path = self.ckpt_dir
        if self.model.is_training and self.cfg.MODEL.WEIGHTS != "":
            print(f"Use {self.cfg.MODEL.WEIGHTS} instead of {ckpt_path}.")
            ckpt_path = self.cfg.MODEL.WEIGHTS
        if sess is None:
            sess = self.sess
        wnn.restore_variables(sess,ckpt_path,**kwargs)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.is_training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        if self.step%self.log_step == 0:
            _,total_loss,self.step,summary = self.sess.run([self.train_op,self.total_loss,self.global_step,self.summary])
            self.summary_writer.add_summary(summary, self.step)
        elif self.full_trace:
            if self.step %3==0:
                print("Full trace tensorflow may cause segment errors.")
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, total_loss, self.step = self.sess.run([self.train_op, self.total_loss, self.global_step],
                                                     options=run_options, run_metadata=run_metadata)
            self.summary_writer.add_run_metadata(run_metadata, f"setp{self.step}")
        else:
            _,total_loss,self.step = self.sess.run([self.train_op,self.total_loss,self.global_step])

        t_cost = time.perf_counter() - start
        print(f"{self.name} step={self.step}, loss={total_loss}, time_cost={t_cost}")

        if self.step%20 == 0:
            sys.stdout.flush()

        if self.step%self.save_step == 0 and self.step>2:
            print("save check point file.")
            self.saver.save(self.sess, os.path.join(self.ckpt_dir,CHECK_POINT_FILE_NAME),global_step=self.step)
        if self.max_train_step>1 and self.step>self.max_train_step:
            self.saver.save(self.sess, os.path.join(self.ckpt_dir,CHECK_POINT_FILE_NAME),global_step=self.step)
            sys.stdout.flush()
            raise Exception("Train Finish")

    def run_eval_step(self):
        assert not self.model.is_training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        if self.step%self.log_step == 0:
            res_data = self.res_data_for_eval
            summary_dict = {"summary":self.summary}
            res_data.update(summary_dict)
            res_data = self.sess.run(res_data)
            self.summary_writer.add_summary(res_data['summary'], self.step)
            sys.stdout.flush()
        elif self.full_trace:
            if self.step % 3 == 0:
                print("Full trace tensorflow may cause segment errors.")
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            res_data = self.res_data_for_eval
            res_data = self.sess.run(res_data,
                                     options=run_options, run_metadata=run_metadata)
            self.summary_writer.add_run_metadata(run_metadata, f"setp{self.step}")
        else:
            res_data = self.res_data_for_eval
            res_data = self.sess.run(res_data)
        self.step += 1

        t_cost = time.perf_counter() - start
        print(f"{self.name} step={self.step}, time_cost={t_cost}")

        return res_data


    @classmethod
    def build_model(cls, cfg,**kwargs):
        """
        Returns:

        It now calls :func:`object_detection2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg,**kwargs)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model
