from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.trainers import Trainer, TripTrainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.sampler import RandomIdentitySampler, RandomNonPairSampler
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

def get_data(name, split_id, data_dir, height, width, batch_size, workers,
             combine_trainval, num_instances, repeat_times):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(dataset.train, root=osp.join(dataset.images_dir,dataset.train_path),
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
	sampler=RandomNonPairSampler(dataset.train, num_instances = num_instances, repeat_times = repeat_times, batch_size = batch_size),
        pin_memory=True, drop_last=True)

    query_loader = DataLoader(
        Preprocessor(dataset.query, root=osp.join(dataset.images_dir,dataset.query_path),
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=osp.join(dataset.images_dir,dataset.gallery_path),
                    transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, query_loader, gallery_loader


def main(args):

    feature_save = True
    if args.record_dir:
        writer = SummaryWriter(comment = "New Test", log_dir = args.record_dir)
    else:
        writer = False


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)
    dataset, num_classes, train_loader, query_loader, gallery_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers,
                 args.combine_trainval, num_instances = args.num_instances,
                 repeat_times = args.repeat)

    # Create model
    print('args.margin ', args.margin)
    print('args.trip_weight', args.trip_weight)
    print('before creation')
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes,feat_save=feature_save)
    print('after creation')

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    model = nn.DataParallel(model).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    # print('Evaluator')
    evaluator = Evaluator(model, writer)
    if args.evaluate:
        #metric.train(model, train_loader)
        #print("Validation:")
        #evaluator.evaluate(val_loader, dataset.val, dataset.val, metric)
        print("Test:")
        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)
        return

    # Criterion
    # print('Criterion')
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer
    # print('Optimizer')
    if hasattr(model.module, 'base'):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Trainer
    # print('Trainer')
    trainer = TripTrainer(model, criterion, margin = args.margin, 
						trip_weight = args.trip_weight, 
						sample_strategy = args.sample, 
						dice = args.dice, writer = writer,
                        same_camera_check = args.same_cam_check)

    # Schedule learning rate
    print('Schedule learning rate')
    def adjust_lr(epoch):
        step_size = 60 if args.arch == 'inception' else 40
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    print('Start training')
    for epoch in range(start_epoch, args.epochs):
        #initial learning rate 0.001
        #decay to 0.0001 after 40 epochs
        
        adjust_lr(epoch)
           
        #modified by hht to satisfy learning rate change
        """
        lr = args.lr if epoch < 40 else 0.1 * args.lr
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
        """
        
        trainer.train(epoch, train_loader, optimizer)        
        is_best = True

        
        if epoch < args.start_save:
            continue
        top1 = evaluator.evaluate(query_loader, gallery_loader, 
								dataset.query, dataset.gallery, 
								(epoch + 1, len(train_loader)))

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        
        if epoch % 10 == 0:
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1
            }, is_best, fpath=osp.join(args.logs_dir, '%d_checkpoint.pth.tar' % (epoch))) 

        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    metric.train(model, train_loader)
    evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    #newly added arguments
    parser.add_argument('--margin', type=float, default=2.0)
    parser.add_argument('--trip_weight', type=float, default=0.5)
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=80)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--dice', type=int, default=0)
    #metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--record-dir', type=str, metavar='PATH',
						default='')
    parser.add_argument('--same_cam_check', type=str, default='', metavar='PATH')
    main(parser.parse_args())
