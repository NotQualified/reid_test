from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomNonPairSampler(Sampler):
    
    def __init__(self, data_source, num_instances = 1, batch_size = 256, repeat_times = 80):
        self.data_source = data_source
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.rindex_dic = defaultdict(list)
        self.gindex_dic = defaultdict(list)
        for index, (fname, pid, _) in enumerate(data_source):
            fname = fname.replace(".jpg", "")
            if 'g' in fname:
                self.gindex_dic[pid].append(index)
            else:
                self.rindex_dic[pid].append(index)
        #print(self.rindex_dic)
        #print(self.gindex_dic)
        self.pids = list(self.rindex_dic.keys())
        self.num_samples = len(self.pids)
        self.repeat_times = repeat_times

    def __len__(self):
    	return self.batch_size * self.repeat_times

    def __iter__(self):
        ret = []
        for _ in range(self.repeat_times):
            query = torch.randperm(self.num_samples)
            for i in range(self.batch_size // (2 * self.num_instances)):
                #print(len(self.pids))
                pid = self.pids[query[i]]
		# len(rt) = len(gt)
                rt = self.rindex_dic[pid]
                gt = self.gindex_dic[pid]
                if(len(rt) >= self.num_instances):
                    rt = np.random.choice(rt, size = self.num_instances, replace = False)
                    gt = np.random.choice(gt, size = self.num_instances, replace = False)
                else:
                    rt = np.random.choice(rt, size = self.num_instances, replace = True)
                    gt = np.random.choice(gt, size = self.num_instances, replace = True)
                ret.extend(rt)
                ret.extend(gt)
        return iter(ret)
	    

#added by hht
class RandomTwinSampler(Sampler):
    def __init__(self, data_source, batch_size, repeat_time):
        self.data_source = data_source
        self.batch_size = batch_size
        self.index_dic = defaultdict(list)
        for index, temp in enumerate(data_source):
            #temp[0][1] represent pid
            self.index_dic[temp[0][1]].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.batch_size * self.repeat_time

    def __iter__(self):
        sample_dict = defaultdict(list)
        ret = []
        for key in self.index_dic.keys():
            #generate random index for each pid
            l = len(self.index_dic[key])
            sample_dict[key] = torch.randperm(l)
        for _ in repeat_time:
            indices = torch.randperm(self.num_samples)
            #fetch front batch_size element
            #don't use random index to simplify 
            for i in range(0, self.batch_size):
                num = len(self.index_dic[indices[i]])
                rd = np.random.randint(num, 1)[0]
                ret.append(self.index_dic[indices[i]][rd])
        return iter(ret)
    
