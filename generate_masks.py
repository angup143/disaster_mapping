from albumentations import Compose, Normalize
"""
Script generates predictions, splitting original images into tiles, and assembling prediction back together
"""
import argparse
import cv2
import os
import csv
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import utils
from torch.utils.data import DataLoader
from torch.nn import functional as F
import sys
from skimage.morphology import skeletonize


from aerial_dataset import AerialDataset, AerialRoadDataset, AerialCombinedDataset
from models import UNet16, LinkNet34, UNet11, UNet,  LinkNet18, UNet16Upsample, UNet11Upsample, UNet18, UNet34, UNet18Upsample, UNet34Upsample

h_start = 0
w_start = 0
binary_factor = 100

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


def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)


def get_model(model_path, model_type, num_classes):
    """

    :param model_path:
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34',
    :param problem_type: 'binary', 'parts', 'instruments'
    :return:
    """

    if model_type == 'UNet':
        model = UNet(num_classes=num_classes)
    else:
        model_name = model_list[model_type]
        model = model_name(num_classes=num_classes)
#    print(model)
    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key,
             value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model


def predict(model, filepath, batch_size, to_path, img_transform, datatype, img_size, num_classes):
    if datatype == 'buildings':
        loader = DataLoader(
            dataset=AerialDataset(
                filepath, transform=img_transform, mode='predict'),
            shuffle=False,
            batch_size=batch_size,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available()
        )
    elif datatype == 'combined':
        loader = DataLoader(
            dataset=AerialCombinedDataset(
                filepath, transform=img_transform, mode='predict'),
            shuffle=False,
            batch_size=batch_size,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        loader = DataLoader(
            dataset=AerialRoadDataset(
                filepath, transform=img_transform, mode='predict'),
            shuffle=False,
            batch_size=batch_size,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available())

    with torch.no_grad():
        model.eval()
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
            inputs = utils.cuda(inputs)
            outputs = model(inputs)
            print('output_shape', np.shape(outputs))
            for i, image_name in enumerate(paths):
                factor = binary_factor
                if datatype == 'combined' or num_classes > 1:
                    output_classes = outputs.data.cpu().numpy().argmax(axis=1)
                    t_mask = (output_classes[i]*100).astype(np.float64)

                else:
                  #  t_mask = (F.sigmoid(outputs[i, 0]).data.cpu().numpy() * factor).astype(np.float64)
                    t_mask = ((outputs[i, 0] > 0).data.cpu(
                    ).numpy() * factor).astype(np.float64)
                    confidence_mask = outputs[i, 0].data.cpu(
                    ).numpy().astype(np.float64)
                    confidence_mask = np.repeat(
                        confidence_mask[:, :, np.newaxis], 3, axis=2)
                    confidence_mask = cv2.normalize(
                        confidence_mask,  confidence_mask, 0, 255, cv2.NORM_MINMAX)
                    instrument_folder = Path(paths[i]).parent.parent.name
                    cv2.imwrite(str(to_path / instrument_folder /
                                    (Path(paths[i]).stem + '_confidence.png')), confidence_mask)

                h, w = t_mask.shape
                input_image = cv2.imread(image_name).astype(np.float64)
                full_mask = np.zeros((img_size, img_size))
                full_mask[h_start:h_start + h, w_start:w_start + w] = t_mask
                full_mask = np.repeat(full_mask[:, :, np.newaxis], 3, axis=2)

                if datatype == 'combined' or datatype == 'buildings':
                    full_mask = np.zeros((img_size, img_size, 3))
                    full_mask[t_mask == 100, :] = (0, 255, 255)
                    full_mask[t_mask == 200, :] = (255, 0, 0)

                elif datatype == 'roads':
                    full_mask = np.zeros((img_size, img_size, 3))
                    full_mask[t_mask == 100, :] = (255, 0, 0)

                masked_image = np.zeros((img_size, img_size, 3))

                instrument_folder = Path(paths[i]).parent.parent.name

                (to_path / instrument_folder).mkdir(exist_ok=True, parents=True)
                # save mask
                cv2.imwrite(str(to_path / instrument_folder /
                                (Path(paths[i]).stem + '.png')), full_mask)

                # showing mask on orig image
                cv2.addWeighted(full_mask, 0.7, input_image,
                                0.3, 0, masked_image)
                cv2.imwrite(str(to_path / instrument_folder /
                                (Path(paths[i]).stem + '_combined.tif')), masked_image)

                if datatype == 'roads':
                    test_mask = t_mask.astype(np.uint8)
                    test_mask[test_mask <= 50] = 0
                    test_mask[test_mask > 50] = 1

                    mask_skeleton_tmp = skeletonize(test_mask)
                    mask_skeleton_tmp = 255 * mask_skeleton_tmp
                    mask_skeleton = np.zeros((img_size, img_size))
                    mask_skeleton[h_start:h_start + h,
                                  w_start:w_start + w] = mask_skeleton_tmp
                    mask_skeleton = np.repeat(np.expand_dims(
                        mask_skeleton, axis=-1), 3, axis=2)
                    skeletonized_image = np.zeros((img_size, img_size, 3))

                    # showing skeleton on orig image
                    cv2.addWeighted(mask_skeleton, 0.5, input_image,
                                    0.5, 0, skeletonized_image)
                    cv2.imwrite(str(to_path / instrument_folder /
                                    (Path(paths[i]).stem + '_skeleton_combined.tif')), skeletonized_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, help='path to actual model')
    arg('--model_type', type=str, default='UNet',
        help='network architecture', choices=model_list.keys())
    arg('--output_path', type=str, help='path to save images', default='1')
    arg('--batch-size', type=int, default=4)
    arg('--filepath', type=str, help='file with testing filenames')
    arg('--workers', type=int, default=12)
    arg('--datatype', type=str, default='buildings',
        choices=['buildings', 'roads', 'combined'])
    arg('--img_size', type=int, default=416)
    arg('--num_classes', type=int, default=3)

    args = parser.parse_args()

    val_filename = args.filepath
    dataset_type = args.filepath.split("/")[-3]

    model = get_model(str(Path(args.model_path)),
                      model_type=args.model_type, num_classes=args.num_classes)

    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    predict(model, val_filename, args.batch_size, output_path,
            img_transform=img_transform(p=1), datatype=args.datatype, img_size=args.img_size, num_classes=args.num_classes)
