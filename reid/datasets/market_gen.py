from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re

from collections import defaultdict
from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class MarketGen(object):

    def __init__(self, root):

        self.images_dir = osp.join(root)
       # print('dir = ', self.images_dir)

        self.train_path = 'bounding_box_train'
        self.gallery_path = 'bounding_box_test'
        self.query_path = 'query'
#        self.camstyle_path = 'bounding_box_train_camstyle'
        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0
        self.load()

    def preprocess(self, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c([g]?\d)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(self.images_dir, path, '*.jpg')))
        #print('fpaths:', glob(osp.join(self.images_dir, path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
           #print(fname)
            try:
                pid, cam = map(int, pattern.search(fname).groups())
            except BaseException:
                group = pattern.search(fname).groups()
                pid = int(group[0])
                cam = int(group[1][1:])
                #print(group)
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append((fname, pid, cam))
        return ret, int(len(all_pids))

    # added by hht in order to generate paired iterable list which elements is [(fname, pid, cam), (fgname, pid, cam)]
    def twin_preprocess(self, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c([g]?\d)')
        all_pids = {}
        twin_dict = defaultdict(list)
        ret = []
        fpaths = sorted(glob(osp.join(self.images_dir, path, '*.jpg')))
        #print('fpaths:', glob(osp.join(self.images_dir, path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            flag = 0
            #print(fname)
            try:
                pid, cam = map(int, pattern.search(fname).groups())
            except BaseException:
                flag = 1
                group = pattern.search(fname).groups()
                pid = int(group[0])
                cam = int(group[1][1:])
                fname = fname.replace("cg", "c")
                #print(group)
            #print((fname, pid, cam))
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            #ret.append((fname, pid, cam))
            #print(twin_dict[fname])
            twin_dict[fname].append((fname, pid, cam))
        for key in twin_dict.keys():
            ret.append(twin_dict[key])
        #print("ret = \n", ret)
        return ret, int(len(all_pids))
    
    def load(self):
        self.train, self.num_train_ids = self.preprocess(self.train_path)
        #print('train images:', len(self.train))
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False)
        self.query, self.num_query_ids = self.preprocess(self.query_path, False)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
