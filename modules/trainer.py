import os
from abc import abstractmethod
import torch.nn as nn
import time
import torch
import pandas as pd
import numpy as np
from numpy import inf
import math
import json
device = torch.device('cuda:0')


#the disease in DDG4MIMIC, For IU-XRAY, this could be different
dis2id = {
    "pneumothorax": 0,
    "atelectasis": 0,
    "edema": 0,
    "pleural effusion": 0,
    "lung opacity": 0,
    "consolidation": 0,
    "pleural thickening": 0,
    "pneumonia": 0,
    "emphysema": 0,
    "fracture": 2,
    "tortuosity of the thoracic aorta": 1,
    "vascular congestion": 1,
    "air collection": 4,
    "cardiomegaly": 1,
    "calcification": 4,
    "enlargement of the cardiac silhouette": 1,
    "hilar congestion": 5,
    "scoliosis": 2,
    "blunting of the costophrenic angle": 0,
    "heart failure": 1,
    "hernia": 3,
    "granuloma": 4,
    "hematoma": 4,
    "pneumomediastinum": 5,
    "contusion": 4,
    "gastric distention": 3,
    "hypoxemia": 4
}

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError



    def get_mask(self,f_read):
        n = len(f_read["nodes"])
        matrix = np.zeros((n, n), dtype=int)
        for disease in f_read["hierarchy"]:
            y = f_read["entity2id"][disease]
            for topic in f_read["hierarchy"][disease]:
                new_topic = topic + "@"
                if new_topic in f_read["nodes"]:
                    x = f_read["entity2id"][new_topic]
                else:
                    x = f_read["entity2id"][topic]
                matrix[x, x] = 1
                matrix[y, y] = 1
                matrix[y, x] = 1

                for des in f_read["hierarchy"][disease][topic]:
                    for ent in f_read["hierarchy"][disease][topic][des]:
                        z = f_read["entity2id"][ent]
                        matrix[x, z] = 1
                        matrix[z, z] = 1
        return torch.tensor(matrix).to(self.device)

    def train(self):


        with open(
                "path of DDG4MIMIC.json",
                "r", encoding="utf-8") as f:
            f_read = json.load(f)

        #get adjacent matrix
        matrix = self.get_mask(f_read)
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch,f_read["nodes"],matrix)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table._append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table._append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def get_embedding(self,nodes):
        embeddings = torch.empty(0,512).to(self.device)
        for node in nodes:
            seq = self.model.tokenizer(node)
            embedding = self.model.encoder_decoder.embedding(torch.tensor(seq[1:-2]).to(self.device))
            embedding = torch.mean(embedding,dim=0,keepdim=True)
            embeddings = torch.cat([embeddings,embedding],dim=0)
        return embeddings


    def _train_epoch(self, epoch,nodes,matrix):
        j = 0
        train_loss = 0
        self.model.train()
        for batch_idx, (image_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks =  images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device)
            j += 1

            # monotonic annealing schedule for MIMIC-CXR
            if epoch == 1:
                mod = int(j / 500)
                modu = (mod + epoch - 1)
                rate = min((modu * 1.0) / 20, 0.4)
            else:
                rate = 0.4

            # monotonic annealing schedule for IU-XRay
            # if epoch <=5:
            #     rate = (epoch-1)/(10*1.0)
            # else:
            #     rate = 0.4

            embedding = self.get_embedding(nodes)
            output,div = self.model(images,embedding, matrix, reports_ids,mode='train')
            b, l, _ = output.size()
            indicator = output[:, 0, :].unsqueeze(dim=1)
            indicator = torch.repeat_interleave(indicator, l - 1, dim=1)

            # BOW planning loss
            loss1 = self.criterion(indicator, reports_ids, reports_masks)
            # generation loss
            loss = self.criterion(output[:, 1:, :], reports_ids, reports_masks)
            # loss = self.criterion(output, reports_ids, reports_masks)

            print(loss)

            loss = loss+ 0.03*loss1 + rate*(0.0008*div)
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            embedding = self.get_embedding(nodes)
            embedding = self.model.encoder_decoder.model.get_node_embedding( embedding, matrix)

            for batch_idx, (
            image_id,  images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks =  images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                output,_ = self.model(images,embedding, matrix, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                print(reports[0])
                val_res.extend(reports)
                val_gts.extend(ground_truths)

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                        {i: [re] for i, re in enumerate(val_res)})

            print('val all metrics:',val_met)
            log.update(**{'val_' + k: v for k, v in val_met.items()})


        self.model.eval()
        with torch.no_grad():
            embedding = self.get_embedding(nodes)
            embedding = self.model.encoder_decoder.model.get_node_embedding( embedding, matrix)
            test_gts, test_res = [], []
            dict = {}
            for batch_idx, (
            image_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks =  images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                output,_ = self.model(images,embedding, matrix, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                for i,id in enumerate(image_id):
                    dict[id] = reports[i]

                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                       {i: [re] for i, re in enumerate(test_res)})
            print('test all metrics:', test_met)

            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        return log

