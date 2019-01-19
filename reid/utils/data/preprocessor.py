from __future__ import absolute_import
import os.path as osp

from PIL import Image


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        #added by hht to add read for pairs images
        #the dataset format changed from :[(img,fname,pid,camid)...]
        #to [[(img1,fname1,pid,camid), (img2,fname2,pid,camid)]...]
        if(isinstance(self.dataset[index], list)):
            #print("istwin")
            twin = []
            for fname, pid, camid in self.dataset[index]:
                fpath = fname
                if self.root is not None:
                    fpath = osp.join(self.root, fname)
                img = Image.open(fpath).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                twin.append((img, fname, pid, camid))
            return twin    
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid
