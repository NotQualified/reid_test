import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MixedLoss(nn.Module):
    def __init__(self, margin = 0, class_weight = 1, trip_weight = 1, num_instances = 1):
        super(MixedLoss, self).__init__()
        self.margin = margin
        self.trip_weight = trip_weight
        self.class_weight = class_weight
        self.trip_loss = nn.MarginRankingLoss(margin = margin)
        self.class_loss = nn.CrossEntropyLoss()
        self.num_instances = num_instances

    def forward(self, inputs, targets):
        #inputs: feature map in (batch_size * num_features) shape, class_result in (batch_size * num_classes) shape
        #targets: target label in (batch_size) shape
        features = inputs
        #print(classes.size(), features.size())
        bs = features.size(0)
        r = [True if (i // (self.num_instances / 2)) % 2 == 0 else False for i in range(bs)]
        g = [False if (i // (self.num_instances / 2)) % 2 == 0 else True for i in range(bs)]
        r = torch.ByteTensor(r).cuda()
        g = torch.ByteTensor(g).cuda()
        distmat = torch.pow(features, 2).sum(1, keepdim = True).expand(bs, bs)
        distmat = distmat + distmat.t()
        distmat = distmat.addmm_(1, -2, features, features.t())
        distmat = distmat.clamp(min = 1e-12).sqrt()
        valid = targets.expand(bs, bs).eq(targets.expand(bs, bs).t()) 
        nvalid = ~valid
        #print(valid)
        #print(nvalid)
        ap, an = [], []
        for i in range(bs):
            #select hardest positive
            ap.append(distmat[i][valid[i]].max().unsqueeze(0))
            #select easist negative
            an.append(distmat[i][nvalid[i]].min().unsqueeze(0))
        ap = torch.cat(ap)
        an = torch.cat(an)
        #print(ap.size())
        #print(an.size())
        y = an.data.new()
        y.resize_as_(an.data)
        y.fill_(1)
        y = Variable(y)
        return self.trip_weight * self.trip_loss(an, ap, y)
           # self.class_weight * self.class_loss(classes, targets) 
