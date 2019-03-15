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
        _, _, features = inputs
        #print(classes.size(), features.size())
        bs = features.size(0)
        r = [True if (i // (self.num_instances / 2)) % 2 == 0 else False for i in range(bs)]
        g = [False if (i // (self.num_instances / 2)) % 2 == 0 else True for i in range(bs)]
        r = torch.ByteTensor(r).cuda()
        g = torch.ByteTensor(g).cuda()
        #print(r)
        #print(g)
        #dist = features.pow(2).sum(1)
        #temp = features.pow(2).sum(1, keepdim=True).expand(-1, bs)
        #print(dist.size(), dist)
        #print(temp.size())
        #distmat = temp + temp.t() - 2 * torch.mm(features, features.t()) 
        distmat = features.pow(2).sum(1, keepdim = True).expand(-1, bs)
        distmat = distmat + distmat.t()
        distmat = distmat - 2 * torch.mm(features, features.t())
        distmat = distmat.clamp(min = 1e-12).sqrt()
        valid = targets.unsqueeze(0).t().expand(-1, bs) == targets.unsqueeze(0).expand(bs, -1)
        nvalid = ~valid
        #print(valid)
        #print(nvalid)
        ap = torch.Tensor().cuda()
        an = torch.Tensor().cuda()
        for i in range(bs):
            #select hardest positive
            dist_ap = max(distmat[i][valid[i]]).unsqueeze(0)
            #select easist negative
            dist_an = min(distmat[i][nvalid[i]]).unsqueeze(0)
            ap = torch.cat((ap, dist_ap), dim = 0)
            an = torch.cat((an, dist_an), dim = 0)
        #y = ap.clone().fill_(1).cuda()
        #print(classes.size(), targets.size())
        #if F.relu(ap + self.margin - an).mean() - self.margin <= 0.001:
        #    print(ap, an)

        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        return self.trip_weight * self.trip_loss(an, ap, y)
           # self.class_weight * self.class_loss(classes, targets) 
