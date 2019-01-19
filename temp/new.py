import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.models import ft_net

from PIL import Image

dataset_dir = '../../rep/data/tempMarket1501v15'
ckpt_dir = '../../rep/spgan/checkpoints/espgan_m2d_lam5/Epoch_(15).ckpt'

model_class_num = 751
class_num = 750
device = torch.device('cuda:1,2,3' if torch.cuda.is_available() else 'cpu')

#form the dataset
dirs = {}
dirs['root'] = dataset_dir
dirs['train'] = os.path.join(dataset_dir, 'train')
dirs['gallery'] = os.path.join(dataset_dir, 'test')
dirs['query'] = os.path.join(dataset_dir, 'query')

crop_size_h = 256
crop_size_w = 128
test_transform = transforms.Compose(
    [transforms.Resize((crop_size_h, crop_size_w), Image.BICUBIC), transforms.ToTensor(),
     transforms.Normalize(mean = [0.5] * 3, std = [0.5] * 3)])

#initialize IDE and load_state
IDE = ft_net(model_class_num).to(device)
ckpt = torch.load(ckpt_dir, map_location = device)
IDE.load_state_dict(ckpt['IDE'], strict = False)

query_data = datasets.ImageFolder(dirs['query'], transform = test_transform)
query_loader = data.DataLoader(query_data, batch_size = 1, shuffle = True, num_workers = 0)
gallery_data = datasets.ImageFolder(dirs['gallery'], transform = test_transform)
gallery_loader = data.DataLoader(gallery_data, batch_size = 1, shuffle = True, num_workers = 0)


#calculate features in gallery set
for i, image in enumerate(gallery_loader):
    #IDE.to(device)
    IDE.eval()
    image = image[0].to(device)
    #image = torch.cat((image, image), dim = 0)
    #print(image.size())
    output = IDE.model(image)
    #print(i, output.size())


"""
#iter over query set
for i, image in enumerate(query_loader):
    if i == 0:
        print(i, ':', image[0].size())
"""
