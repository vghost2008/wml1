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

    def __init__(self, cfg,model, data):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__(cfg=cfg)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
        datefmt = "%H:%M:%S")

        self.model = model
        self.data = data
        self.loss_dict = None
        self.sess = None
        self.global_step = tf.train.get_or_create_global_step()
        self.log_step = 1 if model.is_training else 50
        self.save_step = 200
        self.show_eval_results_step = 100
        self.step = 1
        self.total_loss = None
        self.variables_to_train = None
        self.summary = None
        self.summary_writer = None
        self.saver = None
        self.res_data = None
        self.ckpt_dir = os.path.join(self.cfg.ckpt_dir,self.cfg.GLOBAL.PROJ_NAME)
        if model.is_training:
            self.log_dir = os.path.join(self.cfg.log_dir,self.cfg.GLOBAL.PROJ_NAME+"_log")
            self.name = f"{self.cfg.GLOBAL.PROJ_NAME} trainer"
        else:
            self.log_dir = os.path.join(self.cfg.log_dir,self.cfg.GLOBAL.PROJ_NAME+"_eval_log")
            self.name = f"{self.cfg.GLOBAL.PROJ_NAME} evaler"
        self.input_data = None
        print("Log dir",self.log_dir)
        print("ckpt dir",self.ckpt_dir)
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        self.build_net()

        self.res_data_for_eval = self.res_data
        self.res_data_for_eval.update(self.input_data)


    def __del__(self):
        if self.saver is not None:
            self.saver.save(self.sess, os.path.join(self.ckpt_dir, CHECK_POINT_FILE_NAME), global_step=self.step)
        self.sess.close()
        self.summary_writer.close()

    def build_net(self):
        if not os.path.exists(self.log_dir):
            wmlu.create_empty_dir(self.log_dir)
        if not os.path.exists(self.ckpt_dir):
            wmlu.create_empty_dir(self.ckpt_dir)
        data = self.data.get_next()
        DataLoader.detection_image_summary(data,name="data_source")
        self.input_data = data
        if self.cfg.GLOBAL.DEBUG:
            data[IMAGE] = tf.Print(data[IMAGE],[tf.shape(data[IMAGE]),data[ORG_HEIGHT],data[ORG_WIDTH],data[HEIGHT],data[WIDTH]],summarize=100,
                                   name="XXXXX")
        self.res_data,loss_dict = self.model.forward(data)
        if self.model.is_training:
            for v in loss_dict.values():
                tf.summary.scalar(f"loss/{v.name}",v)
                tf.losses.add_loss(v)

        self.loss_dict = loss_dict
        self.sess = tf.Session()

        if self.model.is_training:
            steps = self.cfg.SOLVER.STEPS
            print("Train steps:",steps)
            lr = wnn.build_learning_rate(self.cfg.SOLVER.BASE_LR,global_step=self.global_step,
                                     lr_decay_type="piecewise",steps=steps,decay_factor=0.1,warmup_epochs=0)
            self.max_train_step = steps[-1]
            self.train_op,self.total_loss,self.variables_to_train = wnn.nget_train_op(self.global_step,lr=lr,
                                                                                      clip_norm=self.cfg.SOLVER.CLIP_NORM)
            print("variables to train:")
            wmlu.show_list(self.variables_to_train)
            for v in self.variables_to_train:
                wsummary.histogram_or_scalar(v,v.name)

            self.saver = tf.train.Saver()
            tf.summary.scalar("total_loss",self.total_loss)

        self.summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("Update ops.")
        wmlu.show_list([x.name for x in tf.get_collection(tf.GraphKeys.UPDATE_OPS)])

    def resume_or_load(self,ckpt_path=None,*args,**kwargs):
        if ckpt_path is None:
            ckpt_path = self.ckpt_dir
        wnn.restore_variables(self.sess,ckpt_path)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.is_training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """

        """
        If your want to do something with the losses, you can wrap the model.
        """
        if self.step%self.log_step == 0:
            _,total_loss,self.step,summary = self.sess.run([self.train_op,self.total_loss,self.global_step,self.summary])
            self.summary_writer.add_summary(summary, self.step)
            sys.stdout.flush()
        else:
            _,total_loss,self.step = self.sess.run([self.train_op,self.total_loss,self.global_step])

        t_cost = time.perf_counter() - start
        print(f"{self.name} step={self.step}, loss={total_loss}, time_cost={t_cost}")

        if self.step%self.save_step == 0:
            print("save check point file.")
            self.saver.save(self.sess, os.path.join(self.ckpt_dir,CHECK_POINT_FILE_NAME),global_step=self.step)
            sys.stdout.flush()
        if self.max_train_step>0 and self.step>self.max_train_step:
            self.saver.save(self.sess, os.path.join(self.ckpt_dir,CHECK_POINT_FILE_NAME),global_step=self.step)
            sys.stdout.flush()
            raise Exception("Train Finish")

    def run_eval_step(self):
        """
        Implement the standard training logic described above.
        """
        assert not self.model.is_training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """

        """
        If your want to do something with the losses, you can wrap the model.
        """
        if self.step%self.log_step == 0:
            res_data = self.res_data_for_eval
            summary_dict = {"summary":self.summary}
            res_data.update(summary_dict)
            res_data = self.sess.run(res_data)
            self.summary_writer.add_summary(res_data['summary'], self.step)
            sys.stdout.flush()
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

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg,**kwargs)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model
