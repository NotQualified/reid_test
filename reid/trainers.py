from __future__ import print_function, absolute_import
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            #print('len(data_loader):', len(data_loader))
            #print('inputs.size():', inputs[0].size())
            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)
        
            if isinstance(targets, tuple):
                targets = torch.cat((targets[0], targets[1]))

            losses.update(loss.data.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            """
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
            """
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    'Loss {:.3f} ({:.3f})\t'
                    .format(epoch, i + 1, len(data_loader),
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg,
                        losses.val, losses.avg))
            """
            if i + 1 >= len(data_loader):
                break
	        """

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, fnames, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        #print('fnames : ', fnames)
        #print('len:', len(fnames))
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        #print(outputs[0].size())
        #print(targets.size())
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec

class TripTrainer(BaseTrainer):

    def __init__(self, model, criterion, margin = 2, trip_weight = 1, sample_strategy = -1, dice = 0, writer = False):
        BaseTrainer.__init__(self, model, criterion)
        self.margin = margin
        self.trip_weight = trip_weight
        self.sample_strategy = sample_strategy
        self.dice = dice
        if writer:
            self.writer = writer

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            #print('len(data_loader):', len(data_loader))
            #print('inputs.size():', inputs[0].size())
            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)
        
            if isinstance(targets, tuple):
                targets = torch.cat((targets[0], targets[1]))
            if isinstance(loss, tuple):
                class_loss, trip_loss = loss
                loss = class_loss + self.trip_weight * trip_loss

            losses.update(loss.data.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                if self.writer:
					self.writer.add_scalars('Train', 
											{'trip_loss': trip_loss.item(), 
											'class_loss': class_loss.item(), 
											'total_loss': loss.item()}, 
											epoch * len(data_loader) + i)
                print('Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    'Loss {:.3f} ({:.3f})\t'
                    .format(epoch, i + 1, len(data_loader),
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg,
                        losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, fnames, pids, _ = inputs
        imgs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return (imgs, fnames, pids), targets

    def _forward(self, inputs, targets):
        imgs, fnames, pids = inputs
        batch_size = len(fnames)
        outputs, feat = self.model(*imgs)
        #cross entropy loss
        class_loss = self.criterion(outputs, targets)
        prec, = accuracy(outputs.data, targets.data)
        prec = prec[0]
        num_instances = 1
        for _ in range(batch_size):
            if(self._isgen(fnames[0]) !=  self._isgen(fnames[num_instances])):
                break
            num_instances += 1
        #print("num_instances :", num_instances)
        anchors = []
        positives = []
        negatives = []
        test = []
        for i in range(batch_size // (num_instances * 2)): 
            for j in range(num_instances):
                t = []
                #totally random
                if self.dice == 0:
                    dice = torch.randn(1).item()
                else:
                    dice = self.dice
                #dice = -1
                rand_seed = 1
                if dice >= 0:
                    anchors.append(feat[i * num_instances * 2 + j])
                    t.append(i * num_instances * 2 + j)
                    positives.append(feat[i * num_instances * 2 + num_instances + j])
                    t.append(i * num_instances * 2 + num_instances + j)
                    if self.sample_strategy == 1:
                        #anchor: real positive: gen, negative: real
                        try:
                            negatives.append(feat[i * num_instances * 2 + j + rand_seed * num_instances * 2])
                            t.append(i * num_instances * 2 + j + rand_seed * num_instances * 2)
                        except BaseException:
                            negatives.append(feat[i * num_instances * 2 + j + rand_seed * num_instances * 2 - batch_size])
                            t.append(i * num_instances * 2 + j + rand_seed * num_instances * 2 - batch_size)
                    else:
                        #anchor: real positive: gen, negative: gen
                        try:
                            negatives.append(feat[i * num_instances * 2 + j + rand_seed * num_instances * 2 + num_instances])
                            t.append(i * num_instances * 2 + j + rand_seed * num_instances * 2 + num_instances)
                        except BaseException:
                            negatives.append(feat[i * num_instances * 2 + j + rand_seed * num_instances * 2 + num_instances - batch_size])
                            t.append(i * num_instances * 2 + j + rand_seed * num_instances * 2 + num_instances - batch_size)
                    
                else:
                    anchors.append(feat[i * num_instances * 2 + num_instances + j])
                    t.append(i * num_instances * 2 + num_instances + j)
                    positives.append(feat[i * num_instances * 2 + j])
                    t.append(i * num_instances * 2 + j)
                    if self.sample_strategy == 1:
                        #anchor: gen positive: real, negative: gen
                        try:
                            negatives.append(feat[i * num_instances * 2 + num_instances + j + rand_seed * num_instances * 2])
                            t.append(i * num_instances * 2 + num_instances + j + rand_seed * num_instances * 2)
                        except BaseException:
                            negatives.append(feat[i * num_instances * 2 + num_instances + j + rand_seed * num_instances * 2 - batch_size])
                            t.append(i * num_instances * 2 + num_instances + j + rand_seed * num_instances * 2 - batch_size)
                    else:
                        #anchor: gen positive: real, negative: real
                        try:
                            negatives.append(feat[i * num_instances * 2 + num_instances + j + rand_seed * num_instances * 2 + num_instances])
                            t.append(i * num_instances * 2 + num_instances + j + rand_seed * num_instances * 2 + num_instances)
                        except BaseException:
                            negatives.append(feat[i * num_instances * 2 + num_instances + j + rand_seed * num_instances * 2 + num_instances - batch_size])
                            t.append(i * num_instances * 2 + num_instances + j + rand_seed * num_instances * 2 + num_instances - batch_size)
                test.append(t)
        #print(pids)
        #print(test)
        
        anchor = torch.Tensor().cuda()
        positive = torch.Tensor().cuda()
        negative = torch.Tensor().cuda()
        for i in range(batch_size // (num_instances * 2)):
            anchor = torch.cat((anchor, torch.unsqueeze(anchors[i], 0)), dim = 0)
            positive = torch.cat((positive, torch.unsqueeze(positives[i], 0)), dim = 0)
            negative = torch.cat((negative, torch.unsqueeze(negatives[i], 0)), dim = 0)
        anchors = anchor
        positives = positive
        negatives = negative
        
        #positives = torch.cat(positives, dim = 0)
        #negatives = torch.cat(negatives, dim = 0)
        #print(anchors.size())
        #print(positives.size())
        #print(negatives.size())
        """
        criterion = nn.TripletMarginLoss(margin = self.margin)
        trip_loss = criterion(anchors, positives, negatives)
        if trip_loss > 0:
            print('trip_loss:', trip_loss)
        """
        trip_loss = self._triplet(anchors, positives, negatives, self.margin)
        print('trip_loss:', trip_loss)
        return (class_loss, trip_loss), prec
    
    def _isgen(self, fname):
        fname = fname.replace("jpg", "")
        return 'g' in fname 

    #reference: siamese-triplet
    
    def _triplet(self, anchors, positives, negatives, margin):
        distance_positive = (anchors - positives).pow(2).sum(1)
        distance_negative = (anchors - negatives).pow(2).sum(1)
        #print(distance_positive - distance_negative)
        losses = F.relu(distance_positive - distance_negative + margin)
        return losses.mean()
"""
    def _triplet(self, anchors, positives, negatives, margin):
        #supposed input size: (N, H) 
        criterion = torch.nn.MSELoss()
        loss = 0
        num_samples = len(anchors)
        for i in range(0, num_samples):
            anchor = anchors[i]
            positive = positives[i]
            negative = negatives[i]
            #print('anchor:', anchor)
            #print('positive', positive)
            #print('negative', negative)
            print(criterion(anchor, positive))
            loss += (criterion(anchor, positive) - criterion(anchor, negative)) / num_samples
        loss += margin
        loss = loss if loss > 0 else 0
        if loss > 0:
            print('trip loss:', loss)
        return loss
"""

#added by hht 
#expect input batch of [(img1, fname1, pid, cam), (img2, fname2, pid, cam)]

class TwinTrainer(BaseTrainer):
    
    
    def _parse_data(self, inputs):
        real, gen = inputs
        real_imgs, _, real_pids, _ = real
        gen_imgs, _, gen_pids, _ = gen
        self.pids = real_pids
        #print('real_pids:', real_pids)
        real_inputs = [Variable(real_imgs)]
        real_targets = Variable(real_pids.cuda())
        gen_inputs = [Variable(gen_imgs)]
        gen_targets = Variable(gen_pids.cuda())
        return (real_inputs, gen_inputs), (real_targets, gen_targets)

    def _forward(self, inputs, targets):
        real_inputs, gen_inputs = inputs
        real_targets, gen_targets = targets
        real_outputs, real_feat = self.model(*real_inputs)
        gen_outputs, gen_feat = self.model(*gen_inputs)
        #print('gen_feat:', gen_feat.size())
        """
        loss: criterion loss + triplet
        """
        #criterion
        batch_size = gen_feat.size()[0]
        new_index = []
        for i in range(0, batch_size):
            while True:
                rand_index = np.random.randint(batch_size)
                if(self.pids[i] != self.pids[rand_index]):
                    new_index.append(rand_index)
                    break

        neg_feat = real_feat[rand_index]
        
        trip_loss = self.triplet(real_feat, gen_feat, neg_feat, 3)

        comb_outputs = torch.cat((real_outputs, gen_outputs))
        comb_targets = torch.cat((real_targets, gen_targets))
        entropy_criterion = torch.nn.CrossEntropyLoss()
        entropy_loss = entropy_criterion(comb_outputs, comb_targets)
        entropy_prec, = accuracy(comb_outputs.data, comb_targets.data)
        #print('entropy_loss:', entropy_loss)
        #print('size:', entropy_loss.size())
        return entropy_loss + 0.5 * trip_loss, entropy_prec

        
    def triplet(self, anchors, positives, negatives, margin):
        criterion  = torch.nn.MSELoss()
        batch_size = anchors.size()[0]
        loss = 0
        for i in range(0, batch_size):
            anchor, positive, negative = anchors[i], positives[i], negatives[i]
            #print(criterion(anchor, positive) - criterion(anchor, negative))
            loss += (criterion(anchor, positive) - criterion(anchor, negative)) / batch_size
        loss += margin
        loss = loss if loss > 0 else 0
        if loss > 0:
            print('trip loss:', loss)
        return loss
