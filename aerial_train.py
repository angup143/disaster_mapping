import argparse
import json
from pathlib import Path
import os
import numpy as np
import time

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from models import UNet11, LinkNet34, UNet, UNet16, LinkNet18, UNet16Upsample, UNet11Upsample, UNet18, UNet34, UNet18Upsample, UNet34Upsample
from validation import validation_binary, validation_multi
from loss import LossBinary, LossMulti
from aerial_dataset import AerialDataset, AerialRoadDataset, AerialISPRSDataset, ACTRoadDataset, AerialCombinedDataset
import utils
import sys


from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop
)

model_list = {'UNet11': UNet11,
              'UNet16': UNet16,
              'UNet': UNet,
              'UNet11Upsample': UNet11Upsample,
              'UNet16Upsample': UNet16Upsample,
              'UNet18Upsample': UNet18Upsample,
              'UNet34Upsample': UNet34Upsample,
              'LinkNet18': LinkNet18,
              'LinkNet34': LinkNet34,
              'UNet18': UNet18,
              'UNet34': UNet34}


def make_loader(file_names, shuffle=False, transform=None, problem_type='binary', batch_size=1, workers=2, datatype='buiildings'):
    if datatype == 'buildings':
        return DataLoader(
            dataset=AerialDataset(
                file_names, transform=transform),
            shuffle=shuffle,
            num_workers=workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )
    elif datatype == 'combined':
        return DataLoader(
            dataset=AerialCombinedDataset(
                file_names, transform=transform),
            shuffle=shuffle,
            num_workers=workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )
    else:
        return DataLoader(
            dataset=AerialRoadDataset(
                file_names, transform=transform),
            shuffle=shuffle,
            num_workers=workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available())


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=0.5, type=float)
    arg('--device-ids', type=str, default='0',
        help='For example 0,1 to run on two GPUs')
    arg('--filepath', type=str, help='folder with images and annotation masks')
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=32)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=12)
    arg('--train_crop_height', type=int, default=416)
    arg('--train_crop_width', type=int, default=416)
    arg('--val_crop_height', type=int, default=416)
    arg('--val_crop_width', type=int, default=416)
    arg('--type', type=str, default='binary', choices=['binary', 'multi'])
    arg('--model', type=str, default='UNet', choices=model_list.keys())
    arg('--datatype', type=str, default='buildings',
        choices=['buildings', 'roads', 'combined'])
    arg('--pretrained', action='store_true',
        help='use pretrained network for initialisation')
    arg('--num_classes', type=int, default=1)

    args = parser.parse_args()

    timestr = time.strftime("%Y%m%d-%H%M%S")

    root = Path(args.root)
    root = Path(os.path.join(root, timestr))
    root.mkdir(exist_ok=True, parents=True)
#    dataset_type = args.filepath.split("/")[-3]
    dataset_type = args.datatype
    print('log', root, dataset_type)
    if not utils.check_crop_size(args.train_crop_height, args.train_crop_width):
        print('Input image sizes should be divisible by 32, but train '
              'crop sizes ({train_crop_height} and {train_crop_width}) '
              'are not.'.format(train_crop_height=args.train_crop_height, train_crop_width=args.train_crop_width))
        sys.exit(0)

    if not utils.check_crop_size(args.val_crop_height, args.val_crop_width):
        print('Input image sizes should be divisible by 32, but validation '
              'crop sizes ({val_crop_height} and {val_crop_width}) '
              'are not.'.format(val_crop_height=args.val_crop_height, val_crop_width=args.val_crop_width))
        sys.exit(0)

    num_classes = args.num_classes

    if args.model == 'UNet':
        model = UNet(num_classes=num_classes)
    else:
        model_name = model_list[args.model]
        model = model_name(num_classes=num_classes, pretrained=args.pretrained)

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        raise SystemError('GPU device not found')

    if args.type == 'binary':
        loss = LossBinary(jaccard_weight=args.jaccard_weight)
    elif args.num_classes == 2:
        labelweights = [89371542, 7083233]
        labelweights = np.sum(labelweights) / \
            (np.multiply(num_classes, labelweights))

        loss = LossMulti(num_classes=num_classes,
                         jaccard_weight=args.jaccard_weight, class_weights=labelweights)

    else:
        #labelweights = [30740321,3046555,1554577]
        #labelweights = labelweights / np.sum(labelweights)
        #labelweights = 1 / np.log(1.2 + labelweights)
        labelweights = [89371542, 29703049, 7083233]
        labelweights = np.sum(labelweights) / \
            (np.multiply(num_classes, labelweights))

        loss = LossMulti(num_classes=num_classes,
                         jaccard_weight=args.jaccard_weight, class_weights=labelweights)

    cudnn.benchmark = True

    train_filename = os.path.join(args.filepath, 'trainval.txt')
    val_filename = os.path.join(args.filepath, 'test.txt')

    def train_transform(p=1):
        return Compose([
            PadIfNeeded(min_height=args.train_crop_height,
                        min_width=args.train_crop_width, p=1),
            RandomCrop(height=args.train_crop_height,
                       width=args.train_crop_width, p=1),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            Normalize(p=1)
        ], p=p)

    def val_transform(p=1):
        return Compose([
            PadIfNeeded(min_height=args.val_crop_height,
                        min_width=args.val_crop_width, p=1),
            CenterCrop(height=args.val_crop_height,
                       width=args.val_crop_width, p=1),
            Normalize(p=1)
        ], p=p)

    train_loader = make_loader(train_filename, shuffle=True, transform=train_transform(
        p=1), problem_type=args.type, batch_size=args.batch_size, datatype=args.datatype)
    valid_loader = make_loader(val_filename, transform=val_transform(p=1), problem_type=args.type,
                               batch_size=len(device_ids), datatype=args.datatype)

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))
    args.root = root
    if args.type == 'binary':
        valid = validation_binary
    else:
        valid = validation_multi

    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        num_classes=num_classes,
        model_name=args.model,
        dataset_type=dataset_type
    )


if __name__ == '__main__':
    main()
