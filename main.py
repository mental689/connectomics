from fusion import *
import torchvision.datasets as dset
from lucchi import LucchiPPDataset, Resize, Scale
from torch.utils import data
import torch
from torch.autograd import Variable
from torchvision import utils as v_utils
from torchvision import transforms
import argparse, os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
try:
    from evaluation.align.jaccard import jaccard_index
except:
    raise ValueError('Download the evaluation package from https://github.com/mental689/evaluation')
from PIL import Image


def parse():
    p = argparse.ArgumentParser('Training Fusion net')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-04)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--root', type=str, default='dataset')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--finetune', action='store_true')
    p.add_argument('--finetune_from', type=str, default='models/fusion_best.pt')
    p.add_argument('--logdir', type=str, default='logs')
    p.add_argument('--dataset', type=str, default='lucchipp')
    p.add_argument('--test', action='store_true')
    return p.parse_args()


def train(args):
    if args.dataset == 'lucchipp':
        img_data = LucchiPPDataset(train=True, transforms=transforms.Compose([
            Resize(size=(args.img_size, args.img_size)),
            Scale()
    ]))
    img_batch = data.DataLoader(img_data, batch_size=args.batch_size,
                                shuffle=True, num_workers=2)
    fusion = nn.DataParallel(FusionGenerator(3, 3, 16))
    if args.cuda:
        fusion.cuda()
    if args.finetune:
        try:
            fusion = torch.load(args.finetune_from)
            print("\n--------model restored--------\n")
        except:
            print("\n--------model not restored--------\n")
            pass

    # loss function & optimizer
    loss_func = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(fusion.parameters(), lr=args.lr)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    writer = SummaryWriter(log_dir=args.logdir)
    # training
    for i in range(args.epochs):
        pbar = tqdm(img_batch)
        num_iter = 0
        for (image, label) in pbar:
            optimizer.zero_grad()
            x = Variable(image)
            y = Variable(label)
            if args.cuda:
                x = x.cuda()
                y = y.cuda()
            pred = fusion(x)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()
            num_iter += 1
            pbar.set_description('Epoch {}, Iter {}, loss: {:.5f}'.format(i+1, num_iter, loss.item()))
            writer.add_scalars(main_tag='Training', tag_scalar_dict={
                'loss': loss.item()
            }, global_step=i*len(img_batch)+num_iter-1)
            if num_iter == len(img_batch):
#                v_utils.save_image(x[0].cpu().data, "./dataset/original_image_{}_{}.png".format(i, num_iter))
#                v_utils.save_image(y[0].cpu().data, "./dataset/label_image_{}_{}.png".format(i, num_iter))
#                v_utils.save_image(pred[0].cpu().data, "./dataset/gen_image_{}_{}.png".format(i, num_iter))
                torch.save(fusion, args.finetune_from)
                writer.add_image(tag='Training orig', img_tensor=x[0], global_step=i*len(img_batch)+num_iter-1)
                writer.add_image(tag='Training label', img_tensor=y[0], global_step=i * len(img_batch) + num_iter - 1)
                writer.add_image(tag='Training gen', img_tensor=pred[0], global_step=i * len(img_batch) + num_iter - 1)


def test(args, i=0):
    if args.dataset == 'lucchipp':
        img_data = LucchiPPDataset(train=False, transforms=transforms.Compose([
            Resize(size=(args.img_size, args.img_size)),
            Scale()
    ]))
    img_batch = data.DataLoader(img_data, batch_size=args.batch_size,
                                shuffle=False, num_workers=2)
    fusion = nn.DataParallel(FusionGenerator(3, 3, 16))
    if args.cuda:
        fusion.cuda()
    fusion.train(False)
    fusion.eval()
    try:
        fusion = torch.load(args.finetune_from)
        print("\n--------model restored--------\n")
    except:
        print("\n--------model not restored--------\n")
        pass
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    writer = SummaryWriter(log_dir=args.logdir)
    # testing
    pbar = tqdm(img_batch)
    num_iter = 0
    jaccard = 0.
    scores = []
    for (image, label) in pbar:
        x = Variable(image)
        y = Variable(label)
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        pred = fusion(x)
        #loss = jaccard_index(pred, y, smooth=100)
        jaccards =  jaccard_index(y.cpu().data.numpy(), pred.cpu().data.numpy())
        scores.extend(jaccards)
        num_iter += 1
        pbar.set_description('Epoch {}, Iter {}, Jaccard Index: {:.5f}'.format(i + 1, num_iter, jaccards.mean()))
    writer.add_scalars(main_tag='Testing', tag_scalar_dict={
        'Jaccard index': np.array(scores).mean()
    }, global_step=i )
    print('Testing Jaccard Index: {}'.format(np.array(scores).mean()))
#    v_utils.save_image(x[0].cpu().data, "./dataset/test_original_image_{}.png".format(i))
#    v_utils.save_image(y[0].cpu().data, "./dataset/test_label_image_{}.png".format(i))
#    v_utils.save_image(pred[0].cpu().data, "./dataset/test_gen_image_{}.png".format(i))
    writer.add_image(tag='Testing orig', img_tensor=x[0], global_step=i)
    writer.add_image(tag='Testing label', img_tensor=y[0], global_step=i)
    writer.add_image(tag='Testing gen', img_tensor=pred[0], global_step=i)


if __name__ == '__main__':
    args = parse()
    if not args.test:
        train(args)
    test(args, i=0)

