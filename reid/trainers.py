from __future__ import print_function, absolute_import
import time
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss, MixedLoss
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
        outputs = self.model(*inputs, True)
        if isinstance(outputs, tuple):
            # now is using class layer
            #o1, o2, o3 = outputs
            #print('outputs',outputs)
            #print('outputs1',outputs[0].shape)
            #print('output2',outputs[1].shape)
            #print('output3',outputs[2].shape)
            #print(o1.size(), o2.size(), o3.size())
            outputs = outputs[2]
            #print('outputs',outputs.shape)
            #print(outputs.size())
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            #print('output.size()', outputs.size())
            #print('target.size()', targets.size())
            #print(targets)
            #print(outputs)
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec

class TripTrainer(BaseTrainer):

    def __init__(self, model, criterion, margin = 2, class_weight = 1, trip_weight = 1, sample_strategy = -1, dice = 0, writer = None, same_camera_check = False):
        BaseTrainer.__init__(self, model, criterion)
        self.margin = margin
        self.class_weight = class_weight
        self.trip_weight = trip_weight
        self.sample_strategy = sample_strategy
        self.dice = dice
        self.same_cam_check = same_camera_check
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
                loss = self.class_weight * class_loss + self.trip_weight * trip_loss

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
        camera_check = self._same_camera_check(fnames)
        si = num_instances
        bi = num_instances * 2
        for i in range(batch_size // (num_instances * 2)):
            sft = bi * i
            nsft = bi * i + si
            for j in range(num_instances):
                if self.same_cam_check and not camera_check[sft + j]:
                    #exclude same camera case
                    print(self.same_cam_check)
                    continue
                t = []
                #totally random
                if self.dice == 0:
                    dice = torch.randn(1).item()
                else:
                    dice = self.dice
                rand_seed = 1
                rsft = bi * rand_seed
                if dice >= 0:
                    anchors.append(feat[sft + j])
                    t.append(sft + j)
                    positives.append(feat[nsft + j])
                    t.append(nsft + j)
                    if self.sample_strategy == 1:
                        #anchor: real positive: gen, negative: real
                        neg_index = sft + j + rsft if sft + j + rsft < batch_size else sft + j + rsft - batch_size
                    else:
                        #anchor: real positive: gen, negative: gen
                        neg_index = sft + j + rsft + si if sft + j + rsft + si < batch_size else sft + j + rsft + si - batch_size      
                    neg_appendence = feat[neg_index]
                else:
                    anchors.append(feat[nsft + j])
                    t.append(nsft + j)
                    positives.append(feat[sft + j])
                    t.append(sft + j)
                    if self.sample_strategy == 1:
                        #anchor: gen positive: real, negative: gen
                        neg_index = nsft + j + rsft if nsft + j + rsft < batch_size else nsft + j + rsft - batch_size
                    else:
                        #anchor: gen positive: real, negative: real
                        neg_index = nsft + j + rsft + si if nsft + j + rsft + si < batch_size else nsft + j + rsft + si - batch_size
                    neg_appendence = feat[neg_index]
                t.append(neg_index)
                negatives.append(neg_appendence)
                test.append(t)
        #print(pids)
        #print(fnames)
        #print(test)
        #print(len(anchors), len(negatives))
        anchor = torch.Tensor().cuda()
        positive = torch.Tensor().cuda()
        negative = torch.Tensor().cuda()
        for i in range(len(anchors)):
            anchor = torch.cat((anchor, torch.unsqueeze(anchors[i], 0)), dim = 0)
            positive = torch.cat((positive, torch.unsqueeze(positives[i], 0)), dim = 0)
            negative = torch.cat((negative, torch.unsqueeze(negatives[i], 0)), dim = 0)
        anchors = anchor
        positives = positive
        negatives = negative
        #print(anchor.size(), negative.size())
        
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

    def _same_camera_check(self, fnames):
        batch_size = len(fnames)
        for i, name in enumerate(fnames[1: ]):
            if self._isgen(fnames[0]) != self._isgen(name):
                num_instances = i + 1
                break
        ret = [True for _ in range(batch_size)]
        pattern1 = re.compile('c[g]?\d+s')
        pattern2 = re.compile('\d+')
        factor = num_instances * 2
        for i in range(batch_size // factor):
            indexes = [j for j in range(i * factor, i * factor + num_instances)]
            for index, name1, name2 in zip(indexes, fnames[i * factor: i * factor + num_instances], fnames[i * factor + num_instances: (i + 1) * factor]):
                temp1 = pattern1.findall(name1)[0]
                cam1 = int(pattern2.findall(temp1)[0])
                temp2 = pattern1.findall(name2)[0]
                cam2 = int(pattern2.findall(temp2)[0])
                #print(temp1, temp2, name1, name2, cam1, cam2)
                ret[index], ret[index + num_instances] = (True, True) if cam1 != cam2 else (False, False)
        #print(ret)
        return ret
    
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

class MixedTrainer(BaseTrainer):
    def __init__(self, model, criterion):
        BaseTrainer.__init__(self, model, criterion)

    def _parse_data(self, inputs):
        imgs, fnames, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets
    
    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        return self.criterion(outputs, targets)

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            #print('len(data_loader):', len(data_loader))
            inputs, targets = self._parse_data(inputs)
            loss = self._forward(inputs, targets)
            #losses.update(loss.data.item(), targets.size(0))
            losses.update(loss.data.item(), targets.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    'Loss {:.3f} ({:.3f})\t'
                    .format(epoch, i + 1, len(data_loader),
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg,
                        losses.val, losses.avg))
