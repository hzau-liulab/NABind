import logging
import math
import os
import shutil
from time import strftime
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR

from utils import initialize_logger
from utils import load_model_from_state_dict
from utils import fine_tune_layer

from performance import PerformanceMetrics

import dgl

# from torchvision.ops import sigmoid_focal_loss

logger = logging.getLogger("selene")


def _metrics_logger(name, out_filepath):
    logger = logging.getLogger("{0}".format(name))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    file_handle = logging.FileHandler(
        os.path.join(out_filepath, "{0}.txt".format(name)))
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)
    return logger


class TrainModel(object):

    def __init__(self,
                 model,
                 data_sampler,
                 loss_criterion,
                 optimizer_class,
                 optimizer_kwargs,
                 batch_size,
                 max_steps,
                 report_stats_every_n_steps,
                 output_dir,
                 save_checkpoint_every_n_steps=1000,
                 save_new_checkpoints_after_n_steps=None,
                 report_gt_feature_n_positives=10,
                 n_validation_samples=None,
                 n_test_samples=None,
                 cpu_n_threads=1,
                 use_cuda=False,
                 data_parallel=False,
                 logging_verbosity=2,
                 checkpoint_resume=None,
                 fine_tune_layers=None,
                 metrics=['cutoff','recall','precision','tnr',
                          'acc','mcc','auc','prc']):
        
        self.model = model
        self.sampler = data_sampler
        self.criterion = loss_criterion
        
        self.optimizer = optimizer_class(
            self.model.parameters(), **optimizer_kwargs)
        
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.nth_step_report_stats = report_stats_every_n_steps
        self.nth_step_save_checkpoint = None
        if not save_checkpoint_every_n_steps:
            self.nth_step_save_checkpoint = report_stats_every_n_steps
        else:
            self.nth_step_save_checkpoint = save_checkpoint_every_n_steps

        self.save_new_checkpoints = save_new_checkpoints_after_n_steps

        logger.info("Training parameters set: batch size {0}, "
                    "number of steps per 'epoch': {1}, "
                    "maximum number of steps: {2}".format(
                        self.batch_size,
                        self.nth_step_report_stats,
                        self.max_steps))

        torch.set_num_threads(cpu_n_threads)

        self.use_cuda = use_cuda
        self.data_parallel = data_parallel

        if self.data_parallel:
            self.model = nn.DataParallel(model)
            logger.debug("Wrapped model in DataParallel")

        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()
            logger.debug("Set modules to use CUDA")

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir+'/para/',exist_ok=True)
        if not os.path.exists(self.output_dir+'/para/paras0.pkl'):
            torch.save(self.model.state_dict(),self.output_dir+'/para/paras0.pkl')####
        
        os.makedirs(self.output_dir+'/feature/',exist_ok=True)####
        
        initialize_logger(
            os.path.join(self.output_dir, "{0}.log".format(__name__)),
            verbosity=logging_verbosity)

        self._test_data=None
        self._validation_data=None
        self._train_data=None
        self._test_metrics = PerformanceMetrics(self.output_dir,metrics=metrics)
        self._validation_metrics = PerformanceMetrics(self.output_dir,metrics=metrics)
        self._train_metrics = PerformanceMetrics(self.output_dir,metrics=metrics)
        self.n_validation_samples=n_validation_samples
        # self.create_train_set()###############
        
        # self._create_validation_set(n_samples=n_validation_samples)

        self._start_step = 0
        self._min_loss = float("inf")
        
        if checkpoint_resume is not None:
            checkpoint = torch.load(
                checkpoint_resume,
                map_location=lambda storage, location: storage)

            self.model = load_model_from_state_dict(
                checkpoint["state_dict"], self.model)

            self._start_step = checkpoint["step"]
            if self._start_step >= self.max_steps:
                self.max_steps += self._start_step

            self._min_loss = checkpoint["min_loss"]
            self.optimizer.load_state_dict(
                checkpoint["optimizer"])
            
            # self.model=fine_tune_layer(self.model,fine_tune_layers)
            # self.optimizer = optimizer_class(self.model.parameters(),**optimizer_kwargs)
            
            if self.use_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

            logger.info(
                ("Resuming from checkpoint: step {0}, min loss {1}").format(
                    self._start_step, self._min_loss))
        
        self._train_logger = _metrics_logger(
                "{0}.train".format(__name__), self.output_dir)
        self._validation_logger = _metrics_logger(
                "{0}.validation".format(__name__), self.output_dir)

        self._train_logger.info("loss")
        self._validation_logger.info("\t".join(["loss"] +
            self._validation_metrics.metrics_list))


    def _create_validation_set(self, n_samples=None):
        logger.info("Creating validation dataset.")
        t_i = time()
        self._validation_data, self._all_validation_targets, self._all_validation_tags, = \
            self.sampler.get_validation_set(self.batch_size)
        t_f = time()
        logger.info(("{0} s to load {1} validation examples ({2} validation "
                     "batches) to evaluate after each training step.").format(
                      t_f - t_i,
                      len(self._validation_data) * self.batch_size,
                      len(self._validation_data)))

    def create_test_set(self):
        logger.info("Creating test dataset.")
        t_i = time()
        self._test_data, self._all_test_targets, self._all_test_tags, = \
            self.sampler.get_test_set(self.batch_size)
        t_f = time()
        logger.info(("{0} s to load {1} test examples ({2} test batches) "
                     "to evaluate after all training steps.").format(
                      t_f - t_i,
                      len(self._test_data) * self.batch_size,
                      len(self._test_data)))
    
    def create_train_set(self):
        logger.info("Creating train dataset.")
        t_i = time()
        self._train_data, self._all_train_targets, self._all_train_tags, = \
            self.sampler.get_train_set(self.batch_size)
        t_f = time()
        logger.info(("{0} s to load {1} test examples ({2} test batches) "
                     "to evaluate after all training steps.").format(
                      t_f - t_i,
                      len(self._train_data) * self.batch_size,
                      len(self._train_data)))

    def _get_batch(self,sample_weight=None):
        t_i_sampling = time()
        batch_sequences,adj,batch_targets,batch_tags, \
                    =self.sampler.sample(batch_size=self.batch_size,sample_weight=sample_weight)
        t_f_sampling = time()
        logger.debug(
            ("[BATCH] Time to sample {0} examples: {1} s.").format(
                 self.batch_size,
                 t_f_sampling - t_i_sampling))
        return (batch_sequences,adj,batch_targets,batch_tags,)

    def train_and_validate(self):
        """
        Trains the model and measures validation performance.

        """
        min_loss = self._min_loss
        # scheduler = ReduceLROnPlateau(
        #     self.optimizer, 'min', patience=16, verbose=True,
        #     factor=0.8)
        # scheduler=CosineAnnealingLR(self.optimizer,self.max_steps,)
        
        sample_weight=None
        
        time_per_step = []
        for step in range(self._start_step, self.max_steps):
            t_i = time()
            train_loss = self.train(sample_weight)
            t_f = time()
            time_per_step.append(t_f - t_i)
            
            # if step%500==0:
            #     sample_weight=self.train_evaluate()
            

            if step % self.nth_step_save_checkpoint == 0:
                checkpoint_dict = {
                    "step": step,
                    "arch": self.model.__class__.__name__,
                    "state_dict": self.model.state_dict(),
                    "min_loss": min_loss,
                    "optimizer": self.optimizer.state_dict()
                }
                
                if self.save_new_checkpoints is not None and \
                        self.save_new_checkpoints >= step:
                    checkpoint_filename = "checkpoint-{0}".format(
                        strftime("%m%d%H%M%S"))
                    self._save_checkpoint(
                        checkpoint_dict, False, filename=checkpoint_filename)
                    logger.debug("Saving checkpoint `{0}.pth.tar`".format(
                        checkpoint_filename))
                
                else:
                    self._save_checkpoint(checkpoint_dict, False)
                
                
                # trainloss=self.train_evaluate('TrainLoss')
                # with open(self.output_dir+'/trainloss.txt','a') as f:
                #     f.write('{}\n'.format(trainloss))

            if step and step % self.nth_step_report_stats == 0:
                logger.info(("[STEP {0}] average number "
                             "of steps per second: {1:.1f}").format(
                    step, 1. / np.average(time_per_step)))
                time_per_step = []
                valid_scores = self.validate()
                validation_loss = valid_scores["loss"]
                self._train_logger.info(train_loss)
                to_log = [str(validation_loss)]
                for k in self._validation_metrics.metrics_list:
                    if k in valid_scores:
                        to_log.append(str(valid_scores[k]))
                    else:
                        to_log.append("NA")
                self._validation_logger.info("\t".join(to_log))
                # scheduler.step(math.ceil(validation_loss * 1000.0) / 1000.0)
                # scheduler.step()
                
                # torch.save(self.model.state_dict(),self.output_dir+'/para/paras_validate'+str(step+1)+'.pkl')
                
                if validation_loss < min_loss:
                    min_loss = validation_loss
                    self._save_checkpoint({
                        "step": step,
                        "arch": self.model.__class__.__name__,
                        "state_dict": self.model.state_dict(),
                        "min_loss": min_loss,
                        "optimizer": self.optimizer.state_dict()}, True)
                    logger.debug("Updating `best_model.pth.tar`")
                logger.info("training loss: {0}".format(train_loss))
                logger.info("validation loss: {0}".format(validation_loss))
            
            with open(self.output_dir+'/para_lr.txt','a') as f:
                f.write(str(self.optimizer.state_dict()['param_groups'][0]['lr'])+'\n')
                
            # torch.cuda.empty_cache()


    def train(self,sample_weight=None):
        self.sampler.set_mode('train')
        self.model.train()
        inputs,adj,targets,tags \
                =self._get_batch(sample_weight)

        inputs = torch.Tensor(inputs)
        # inputs_mesh=torch.Tensor(inputs_mesh)
        targets=torch.Tensor(targets)
        # norms=torch.Tensor(adj[1])

        if self.use_cuda:
            inputs = inputs.cuda()
            # inputs_mesh=inputs_mesh.cuda()
            targets=targets.cuda()
            adj=adj.to('cuda')
            # adj=[g.to('cuda') for g in adj]
            # norms=norms.cuda()
        
        inputs = Variable(inputs)
        # inputs_mesh=Variable(inputs_mesh)
        targets=Variable(targets)
        # norms=Variable(norms)
        
        predictions = self.model(inputs,adj)

        loss=self.criterion(predictions,targets,)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _evaluate_on_data(self, data_in_batches, mode='validate'):
        self.model.eval()

        batch_losses = []
        all_predictions = []

        for (inputs,adj,targets) in data_in_batches:
            inputs = torch.Tensor(inputs)
            # inputs_mesh=torch.Tensor(inputs_mesh)
            targets = torch.Tensor(targets)
            # norms=torch.Tensor(adj[1])

            if self.use_cuda:
                inputs = inputs.cuda()
                # inputs_mesh=inputs_mesh.cuda()
                adj=adj.to('cuda')
                # adj=[g.to('cuda') for g in adj]
                targets = targets.cuda()
                # norms=norms.cuda()

            with torch.no_grad():
                inputs = Variable(inputs)
                # inputs_mesh=Variable(inputs_mesh)
                targets = Variable(targets)
                # norms=Variable(norms)

                predictions = self.model(inputs,adj)
                loss=self.criterion(predictions,targets,)

                all_predictions.append(predictions.data.cpu().numpy())

                batch_losses.append(loss.item())
                
        all_predictions = np.vstack(all_predictions)
        
        if mode=='train_evaluate':
            return batch_losses
        
        return np.average(batch_losses),all_predictions

    def validate(self,byone=False):
        if self._validation_data is None:
            self._create_validation_set(n_samples=self.n_validation_samples)
        
        average_loss,all_predictions=self._evaluate_on_data(
            self._validation_data,mode='validate')
        average_scores = self._validation_metrics.update(all_predictions,
                                                         self._all_validation_targets,
                                                         self._all_validation_tags,
                                                         pair=False)
        
        average_scores["loss"] = average_loss
        
        for name, score in average_scores.items():
            logger.info("validation {0}: {1}".format(name, score))
        
        if byone:
            validate_predictions=os.path.join(
                self.output_dir, "validate_predictions.txt")
            self._validation_metrics.write_predictions_to_file(validate_predictions)
    
            validate_performance = os.path.join(
                self.output_dir, "validate_performance.txt")
            self._validation_metrics.write_feature_scores_to_file(validate_performance)
            
            np.savetxt(self.output_dir+'/feature/validatetags.txt',
                       np.column_stack((self._all_validation_tags,self._all_validation_targets)),
                       fmt='%s',)
        
        return average_scores

    def evaluate(self):
        if self._test_data is None:
            self.create_test_set()
        average_loss,all_predictions=self._evaluate_on_data(
            self._test_data,mode='test')

        average_scores = self._test_metrics.update(all_predictions,
                                                   self._all_test_targets,
                                                   self._all_test_tags,
                                                   pair=False)
        
        average_scores["loss"] = average_loss
        
        test_predictions=os.path.join(
            self.output_dir, "test_predictions.txt")
        self._test_metrics.write_predictions_to_file(test_predictions)

        test_performance = os.path.join(
            self.output_dir, "test_performance.txt")
        self._test_metrics.write_feature_scores_to_file(test_performance)
        
        np.savetxt(self.output_dir+'/feature/testtags.txt',
                   np.column_stack((self._all_test_tags,self._all_test_targets)),
                   fmt='%s',)
        
    def get_train_feature(self):
        pass
    
    def train_evaluate(self,mode):
        """
        mode: str SampleWeight or TrainLoss
        """
        batch_losses=self._evaluate_on_data(self._train_data,mode='train_evaluate')
        if mode=='SampleWeight':
            tags=[x[:6] for x in self._all_train_tags.reshape(-1)]
            tags_unique=list(set(tags))
            tags_unique.sort(key=tags.index)
            return np.column_stack((tags_unique,batch_losses)).tolist()
        elif mode=='TrainLoss':
            return np.average(batch_losses)
        else:
            return 
        
    def _save_checkpoint(self,
                         state,
                         is_best,
                         filename="checkpoint"):
        logger.debug("[TRAIN] {0}: Saving model state to file.".format(
            state["step"]))
        cp_filepath = os.path.join(
            self.output_dir, filename)
        torch.save(state, "{0}.pth.tar".format(cp_filepath))
        if is_best:
            best_filepath = os.path.join(self.output_dir, "best_model")
            shutil.copyfile("{0}.pth.tar".format(cp_filepath),
                            "{0}.pth.tar".format(best_filepath))

