import sys, os
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *
from visdom import Visdom



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'home/vivek16/ICnet/ptsemseg/models/checkpoints/model_best.pth.tar')

def visdom_visual(global_step,loss,win,viz):
    s=np.zeros(1)
    g=np.zeros(1)
    s[0]=global_step
    g[0]=loss
    viz.line(Y=g, X=s,win=win,update='append')

def visdom_visual_accuraccy(global_step,mIoU,win1,viz):
    s=np.zeros(1)
    g=np.zeros(1)
    s[0]=global_step
    g[0]=mIoU
    viz.line(Y=g, X=s,win=win1,update='append')


def train(args):

    # Setup Augmentations
    # data_aug= Compose([RandomRotate(10),
    #                    RandomHorizontallyFlip()])
    data_aug =None



    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=data_aug, img_norm=args.img_norm)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Metrics
    running_metrics = runningScore(n_classes)
        
    # Setup visdom for visualization
    if args.visdom:
        viz = Visdom()
        win = viz.line(Y=np.zeros(1).data, X=np.zeros(1).data, opts=dict(xlabel='global_step', ylabel='loss', title='Training Loss'))
        win1 = viz.line(Y=np.zeros(1).data, X=np.zeros(1).data,opts=dict(xlabel='global_step', ylabel='mIoU', title='mIoU Accuracy'))



    # Setup Model
    model = get_model(args.arch, n_classes)
    
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    
    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)
        optimizer=torch.optim.Adam(model.parameters(), lr=args.l_rate)

    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        loss_fn = cross_entropy2d

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            # optimizer.load_state_dict(checkpoint['optimizer_state'])
            # print("Loaded checkpoint '{}' (epoch {})"
            #       .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 

    best_iou = -100.0 
    for epoch in range(args.n_epoch):
        model.train()
        global_step=0
        for i, (images, labels) in enumerate(trainloader):
            global_step = len(trainloader) * epoch + i

            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            if args.visdom:
                visdom_visual(global_step, loss, win, viz)

            if (i+1) % 1 == 0:
                print("Epoch [%d/%d] Loss: %.4f Iteration: %d" % (epoch+1, args.n_epoch, loss.data[0],i))

        print(">>>>>>>>>>>>>>>>>>>>>EVALUATING>>>>>>>>>>>>>>>>>>>>>>>")
        model.eval()
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            images_val = Variable(images_val.cuda(), volatile=True)
            labels_val = Variable(labels_val.cuda(), volatile=True)

            outputs = model(images_val)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            running_metrics.update(gt, pred)

        score, class_iou = running_metrics.get_scores()
        j=0
        for k,v in score.items():
            print(k, v)
            if (j == 3):
                mIoU = v*100
                visdom_visual_accuraccy(global_step, mIoU, win1, viz)
            j = j + 1
        running_metrics.reset()

        ## Checkpoint Saving  Saving every epoch
        is_best=False

        if (epoch) % 1 == 0:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>Saving CheckPoints>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, is_best)


        if score['Mean IoU : \t'] >= best_iou:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Saving Model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "{}_{}_best_model.pkl".format(args.arch, args.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')

    # Architecture
    parser.add_argument('--arch', nargs='?', type=str, default='icnet',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')

    # Datasets
    parser.add_argument('--dataset', nargs='?', type=str, default='carlascapes',
                        help='Dataset to use [\'carlascapes,pascal, camvid, ade20k etc\']')

    # ROW Size Image
    parser.add_argument('--img_rows', nargs='?', type=int, default=1025,
                        help='Height of the input image')

    # COl Size Image
    parser.add_argument('--img_cols', nargs='?', type=int, default=2049,
                        help='Width of the input image')

    # No of Epochs

    parser.add_argument('--n_epoch', nargs='?', type=int, default=5,
                        help='# of the epochs')
    # Batch Size
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')

    # LR

    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5,
                        help='Learning Rate')

    # Resume
    parser.add_argument('--resume', nargs='?', type=str, default='icnet_cityscapes_trainval_90k.pth',
                        help='Path to previous saved model to restart from')

    # Visdom Visualization

    parser.set_defaults(visdom=True)


    parser.add_argument('--img_norm', dest='img_norm', action='store_true',
                        help='Enable input image scales normalization [0, 1] | True by default')

    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false',
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--feature_scale', nargs='?', type=int, default=0,
                        help='Divider for # of features to use')


    parser.add_argument('--visdom', dest='visdom', action='store_true', 
                        help='Enable visualization(s) on visdom | False by default')

    parser.add_argument('--no-visdom', dest='visdom', action='store_false', 
                        help='Disable visualization(s) on visdom | False by default')

    args = parser.parse_args()
    train(args)
